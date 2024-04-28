import cv2
import argparse
from ultralytics import YOLO
import supervision as sv #must be version 0.3.0
import numpy as np
import datetime
import firebase_admin
import pyrebase
from firebase_admin import credentials
from firebase_admin import firestore
# from google.cloud import storage
import os

cred = credentials.Certificate("sipapd-gemastik-firebase-adminsdk-d9otg-6c108c2572.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
# storage_client = storage.Client()
config={
    "apiKey": "AIzaSyDIL0UFoUPfnbm0m9XbT-UeMLpxXU1FrPs",
    "authDomain": "sipapd-gemastik.firebaseapp.com",
    "projectId": "sipapd-gemastik",
    "storageBucket": "sipapd-gemastik.appspot.com",
    "messagingSenderId": "242023549192",
    "appId": "1:242023549192:web:9028bded288685c34d44bd",
    "measurementId": "G-7WHSF1M6V3",
    "databaseURL": ""
}
firebase = pyrebase.initialize_app(config)
localpath = "detection.jpg"
cloudpath = "detections/detection.jpg"




def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "YOLOv8 Live")
    parser.add_argument(
    "--webcam-resolution", 
    default=(640, 640),
    nargs=2,
    type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    byte_tracker = sv.ByteTrack(match_thresh=0.95, lost_track_buffer = 30)

    cv2.startWindowThread()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model = YOLO("best.pt")
    

    box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
    )

    temp = []

    while True:
        ret, frame = cap.read()
        if ret:
            result = model(frame, conf = 0.4, agnostic_nms = True)[0]
            detections = sv.Detections.from_ultralytics(result)
            # detections = detections[detections.class_id !=0]
            detections = byte_tracker.update_with_detections(detections)
            if len(detections.tracker_id) != 0 and max(detections.tracker_id) >= 10:
                    byte_tracker.reset()
            labels = [
                f"Id:{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id,_
            in detections
        ]

            frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels = labels
            )

            
            if not np.array_equal(temp, detections.tracker_id):
                for x in detections.class_id: 
                    if x in [2,3,4,5] and not np.array_equal(temp, detections.tracker_id):
                        date_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                        cv2.imwrite("detection.jpg", frame)
                        firebase.storage().child(cloudpath).put(localpath)
                        doc_ref = db.collection(u"detections").document(date_time)
                        doc_ref.set({
                            "image_url": firebase.storage().child(cloudpath).get_url(None),
                })
                        # with open("label.txt", "a") as myfile:
                        #     myfile.write(f"bbox: {detections.xyxy}\n conf: {detections.confidence}\n class: {detections.class_id}\n tracker_id: {detections.tracker_id}\n")
                        temp = detections.tracker_id
                        os.remove("detection.jpg")
            
            print(frame)
            cv2.imshow('yolov8', frame)

            if (cv2.waitKey(30) == 27): #escape key
                break

            print(frame.shape)

if __name__ == "__main__":
    main()





