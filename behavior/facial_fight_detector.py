from fer.fer import FER


class FacialFightDetector:
    def __init__(self, angry_threshold=0.60):
        self.emotion_detector = FER(mtcnn=True)
        self.angry_threshold = angry_threshold

    def detect(self, frame):
        emotions = self.emotion_detector.detect_emotions(frame)
        angry_faces = []

        for face in emotions:
            emotion_scores = face.get("emotions", {})
            angry_score = float(emotion_scores.get("angry", 0.0))

            if angry_score >= self.angry_threshold:
                x, y, w, h = face["box"]

                angry_faces.append({
                    "bbox": (x, y, x + w, y + h),
                    "score": angry_score
                })

        return angry_faces