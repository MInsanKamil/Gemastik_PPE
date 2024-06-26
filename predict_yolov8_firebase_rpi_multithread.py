import random
import threading
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
from firebase_admin import messaging
import os
from picamera2 import Picamera2

cred = credentials.Certificate("sipapd-gemastik-firebase-adminsdk-d9otg-f62ecfe6f5.json")
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

class_mapping = {
                0: 'Gloves',
                1: 'Helmet',
                2: 'No-Gloves',
                3: 'No-Helmet',
                4: 'No-Shoes',
                5: 'No-Vest',
                6: 'Shoes',
                7: 'Vest'
            }


def send_to_topic(topic, title, body):
  message = messaging.Message(
    notification=messaging.Notification(title=title, body=body),
    topic=topic
  )
  messaging.send(message)

def upload_to_firebase(frame, detections, date_time_str, class_mapping):
    detections_name = f"detection{random.randint(0, 10000)}.jpg"
    cv2.imwrite(detections_name, frame)
    cloudpath = f"detections/{date_time_str}.jpg"
    firebase.storage().child(cloudpath).put(detections_name)
    doc_ref = db.collection(u"detections").document(date_time_str)
    attribute = [class_mapping[class_id] for class_id in detections.class_id]
    date_time = datetime.datetime.now()
    doc_ref.set({
        "image_url": firebase.storage().child(cloudpath).get_url(None),
        "attribute": attribute,
        "time": date_time,
    })
    pelanggaran = ", ".join(class_mapping[class_id][3:] for class_id in detections.class_id if class_id in [2, 3, 4, 5])
    send_to_topic("detection", "Pelanggaran", f"Terjadi pelanggaran tidak menggunakan {pelanggaran} di lapangan")
    os.remove(detections_name)

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

    cap = Picamera2(0)
    cap.configure(cap.create_preview_configuration(main={"format": "XRGB8888", "size": (frame_width, frame_height)}))
    cap.start()
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model = YOLO("best.onnx")
    

    box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
    )

    temp = []

    while True:
        frame = cap.capture_array()
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame_copy = frame.copy()
        # ret, frame = cap.read()
        # if ret:
        #     frame_copy = frame.copy()
        result = model(frame_copy, conf = 0.4, agnostic_nms = True)[0]
        detections = sv.Detections.from_ultralytics(result)
        # detections = detections[detections.class_id !=0]
        detections = byte_tracker.update_with_detections(detections)
        if len(detections.tracker_id) != 0 and max(detections.tracker_id) >= 100:
                byte_tracker.reset()
        labels = [
            f"Id:{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id,_
        in detections
    ]

        frame = box_annotator.annotate(
        scene=frame_copy,
        detections=detections,
        labels = labels
        )

        
        if not np.array_equal(temp, detections.tracker_id):
            for x in detections.class_id: 
                if x in [2,3,4,5] and not np.array_equal(temp, detections.tracker_id):
                    date_time_str = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
                    threading.Thread(target=upload_to_firebase, args=(frame.copy(), detections, date_time_str, class_mapping)).start()
                    temp = detections.tracker_id
        
        # print(frame)
        # cv2.imshow('yolov8', frame)

        if (cv2.waitKey(30) == 27): #escape key
            break

        # print(frame.shape)

if __name__ == "__main__":
    main()





