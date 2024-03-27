import cv2
import argparse

import ultralytics
import supervision as sv

import time 

from picamera2 import Picamera2

def parse_arguments():
    parser = argparse.ArgumentParser(description = "YOLOv8 Camera")
    parser.add_argument("--webcam-resolution", default = [640, 480], nargs = 2, type = int)
    parser.add_argument("--model-path", default = "best.pt", nargs = 1, type = str)
    return parser.parse_args()

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    model_path = args.model_path

    cv2.startWindowThread()

    camera = Picamera2(0)
    camera.configure(camera.create_preview_configuration(main={"format": "XRGB8888", "size": (frame_width, frame_height)}))
    camera.start()

    model = ultralytics.YOLO(model_path)

    box_annotator = sv.BoxAnnotator(thickness = 2, text_thickness = 2, text_scale = 1)

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        new_frame_time = time.time() 
        frame = camera.capture_array()

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        results = model(frame)[0]

        frame = box_annotator.annotate(frame, detections = results.detections)

        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 

        cv2.putText(frame, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        cv2.imshow("Detections", frame)

        if (cv2.waitKey(30) == 27):
            break

    camera.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()