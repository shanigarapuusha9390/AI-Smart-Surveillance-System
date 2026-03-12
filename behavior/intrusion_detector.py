import cv2
import numpy as np
from config import RESTRICTED_ZONE


class IntrusionDetector:
    def __init__(self, overlap_threshold=0.35):
        self.zone = np.array(RESTRICTED_ZONE, dtype=np.int32)
        self.overlap_threshold = overlap_threshold

    def draw_zone(self, frame):
        cv2.polylines(
            frame,
            [self.zone],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2
        )

        x, y = self.zone[0]
        cv2.putText(
            frame,
            "Restricted Area",
            (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    def check(self, person):
        x1, y1, x2, y2 = map(int, person["bbox"])

        person_w = max(1, x2 - x1)
        person_h = max(1, y2 - y1)
        person_area = person_w * person_h

        max_x = max(x2, int(np.max(self.zone[:, 0]))) + 5
        max_y = max(y2, int(np.max(self.zone[:, 1]))) + 5

        if max_x <= 0 or max_y <= 0:
            person["overlap_ratio"] = 0.0
            return False

        zone_mask = np.zeros((max_y, max_x), dtype=np.uint8)
        cv2.fillPoly(zone_mask, [self.zone], 255)

        person_mask = np.zeros((max_y, max_x), dtype=np.uint8)
        cv2.rectangle(person_mask, (x1, y1), (x2, y2), 255, -1)

        intersection = cv2.bitwise_and(zone_mask, person_mask)
        overlap_area = cv2.countNonZero(intersection)

        overlap_ratio = overlap_area / person_area
        person["overlap_ratio"] = overlap_ratio

        return overlap_ratio >= self.overlap_threshold