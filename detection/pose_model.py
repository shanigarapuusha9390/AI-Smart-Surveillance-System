from ultralytics import YOLO
import numpy as np


class PoseEstimator:
    def __init__(self, model_name="yolov8n-pose.pt"):
        # If this checkpoint name fails in your Ultralytics install,
        # use the pose model name supported by your installed version.
        self.model = YOLO(model_name)

    def infer(self, frame):
        results = self.model(frame, verbose=False)[0]
        outputs = []

        if results.boxes is None or results.keypoints is None:
            return outputs

        boxes = results.boxes.xyxy.cpu().numpy()
        keypoints_xy = results.keypoints.xy.cpu().numpy()

        kp_conf = None
        if getattr(results.keypoints, "conf", None) is not None:
            kp_conf = results.keypoints.conf.cpu().numpy()

        for i in range(len(boxes)):
            item = {
                "bbox": boxes[i].tolist(),
                "keypoints": keypoints_xy[i].tolist(),
            }
            if kp_conf is not None:
                item["kp_conf"] = kp_conf[i].tolist()
            else:
                item["kp_conf"] = [1.0] * len(keypoints_xy[i])

            outputs.append(item)

        return outputs