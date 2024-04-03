import cv2
import argparse
from ultralytics import YOLO
import supervision as sv #must be version 0.3.0
import numpy as np
from picamera2 import Picamera2


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
    model = YOLO("best.pt")
    

    box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
    )

    temp = []

    while True:
        frame = cap.capture_array()
        
        result = model(frame, conf = 0.5, agnostic_nms = True)[0]
        detections = sv.Detections.from_ultralytics(result)
        # detections = detections[detections.class_id !=0]
        detections = byte_tracker.update_with_detections(detections)
        if len(detections.tracker_id) != 0 and max(detections.tracker_id) >= 10:
                byte_tracker.reset()
        if not np.array_equal(temp, detections.tracker_id):
            temp = detections.tracker_id  
            for x in detections.class_id: 
                if x in [2,3,4,5]:
                    with open("label.txt", "a") as myfile:
                        myfile.write(f"bbox: {detections.xyxy}\n conf: {detections.confidence}\n class: {detections.class_id}\n tracker_id: {detections.tracker_id}\n")
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
        
        cv2.imshow('yolov8', frame)

        if (cv2.waitKey(30) == 27): #escape key
            break

        print(frame.shape)

if __name__ == "__main__":
    main()
