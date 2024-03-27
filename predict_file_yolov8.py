import ultralytics

model = ultralytics.YOLO("bird_rusdien_nano.pt")
model.predict("predict.png", save = True, imgsz = 640, conf = 0.5)