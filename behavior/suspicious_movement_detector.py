import math
from collections import defaultdict


class SuspiciousMovementDetector:
    def __init__(self, speed_threshold=25, min_frames=8):
        self.prev_centers = {}
        self.fast_counts = defaultdict(int)
        self.speed_threshold = speed_threshold
        self.min_frames = min_frames

    def detect(self, persons):
        suspicious_ids = []

        for p in persons:
            pid = p["id"]
            x1, y1, x2, y2 = p["bbox"]

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if pid in self.prev_centers:
                px, py = self.prev_centers[pid]
                speed = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

                if speed > self.speed_threshold:
                    self.fast_counts[pid] += 1
                else:
                    self.fast_counts[pid] = max(0, self.fast_counts[pid] - 1)

                if self.fast_counts[pid] >= self.min_frames:
                    suspicious_ids.append(pid)

            self.prev_centers[pid] = (cx, cy)

        return suspicious_ids