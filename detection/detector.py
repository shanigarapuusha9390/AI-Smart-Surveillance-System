from ultralytics import YOLO

class Detector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):

        results = self.model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes.data.tolist():

            x1, y1, x2, y2, score, class_id = box

            # detect only person class
            if int(class_id) != 0:
                continue

            detections.append(
                ([x1, y1, x2 - x1, y2 - y1], score, "person")
            )

        return detections