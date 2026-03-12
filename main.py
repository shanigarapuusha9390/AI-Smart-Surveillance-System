import cv2
import time
import os

from alerts.voice_alert import speak_warning
from alerts.email_alert import send_email_alert
from detection.detector import Detector
from detection.pose_model import PoseEstimator
from tracking.tracker import Tracker
from behavior.intrusion_detector import IntrusionDetector
from behavior.fight_detector import FightDetector
from behavior.suspicious_movement_detector import SuspiciousMovementDetector
from behavior.facial_fight_detector import FacialFightDetector
from buffer.rolling_buffer import RollingBuffer
from alerts.clip_saver import save_clip
from config import FPS, SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL

DASHBOARD_FRAME_DIR = "dashboard_frames"
os.makedirs(DASHBOARD_FRAME_DIR, exist_ok=True)
LIVE_FRAME_PATH = os.path.join(DASHBOARD_FRAME_DIR, "live.jpg")


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


detector = Detector()
pose_estimator = PoseEstimator()
tracker = Tracker()
intrusion = IntrusionDetector(overlap_threshold=0.35)
fight = FightDetector()
suspicious_detector = SuspiciousMovementDetector()
facial_fight = FacialFightDetector()
rolling_buffer = RollingBuffer()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

fight_cooldown = 0
intrusion_cooldown = 0
suspicious_cooldown = 0

last_saved_time = 0
save_gap_seconds = 8

last_voice_time = 0
voice_gap_seconds = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame.")
        break

    rolling_buffer.add(frame)

    detections = detector.detect(frame)
    persons = tracker.update(detections, frame)
    poses = pose_estimator.infer(frame)
    angry_faces = facial_fight.detect(frame)

    intrusion.draw_zone(frame)

    persons_with_pose = []

    for p in persons:
        best_pose = None
        best_iou = 0.0

        for pose in poses:
            score = iou(p["bbox"], pose["bbox"])
            if score > best_iou:
                best_iou = score
                best_pose = pose

        item = dict(p)

        if best_pose is not None and best_iou > 0.2:
            item["keypoints"] = best_pose["keypoints"]
            item["kp_conf"] = best_pose["kp_conf"]
        else:
            item["keypoints"] = None
            item["kp_conf"] = None

        persons_with_pose.append(item)

    fight_alerts = fight.detect(persons_with_pose)
    suspicious_ids = suspicious_detector.detect(persons)

    fight_ids = set()
    for alert in fight_alerts:
        a, b = alert["pair"]
        fight_ids.add(a)
        fight_ids.add(b)

    # If angry face is detected, strengthen fight alert
    angry_face_present = len(angry_faces) > 0
    if angry_face_present and len(persons) >= 1:
        # marks as possible fight support signal
        for p in persons:
            fight_ids.add(p["id"])

    if fight_cooldown > 0:
        fight_cooldown -= 1
    if intrusion_cooldown > 0:
        intrusion_cooldown -= 1
    if suspicious_cooldown > 0:
        suspicious_cooldown -= 1

    any_intrusion = False
    any_fight = len(fight_alerts) > 0 or angry_face_present
    any_suspicious = len(suspicious_ids) > 0

    for p in persons_with_pose:
        x1, y1, x2, y2 = map(int, p["bbox"])
        pid = p["id"]

        is_intrusion = intrusion.check(p)
        is_fight = pid in fight_ids
        is_suspicious = pid in suspicious_ids
        overlap_ratio = p.get("overlap_ratio", 0.0)

        any_intrusion = any_intrusion or is_intrusion

        if is_fight:
            color = (0, 0, 255)
        elif is_intrusion:
            color = (0, 165, 255)
        elif is_suspicious:
            color = (255, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            f"ID {pid}",
            (x1, y1 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Overlap: {overlap_ratio:.2f}",
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        if p.get("keypoints"):
            for kp in p["keypoints"]:
                kx, ky = map(int, kp)
                if kx > 0 or ky > 0:
                    cv2.circle(frame, (kx, ky), 3, (255, 255, 0), -1)

    # Draw angry faces
    for face in angry_faces:
        x1, y1, x2, y2 = face["bbox"]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"ANGRY {face['score']:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    if any_intrusion:
        cv2.rectangle(frame, (20, 10), (420, 70), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "ALERT: INTRUSION",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 165, 255),
            4
        )

    if any_fight:
        cv2.rectangle(frame, (20, 80), (520, 140), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "ALERT: POSSIBLE FIGHT / ANGER",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            4
        )

    if any_suspicious:
        cv2.rectangle(frame, (20, 150), (560, 210), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "ALERT: SUSPICIOUS MOVEMENT",
            (30, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 255),
            4
        )

    now = time.time()

    if any_intrusion and intrusion_cooldown == 0:
        clip_path = None

        if (now - last_saved_time) > save_gap_seconds:
            clip_path = save_clip(
                rolling_buffer.get(),
                alert_type="intrusion",
                fps=FPS
            )
            print("Intrusion clip saved:", clip_path)
            last_saved_time = now

        if (now - last_voice_time) > voice_gap_seconds:
            speak_warning("Warning! Restricted area entered.")
            last_voice_time = now

        send_email_alert(
            subject="ALERT: Intrusion Detected",
            body=f"Intrusion detected by surveillance system.\nSaved clip: {clip_path}",
            sender_email=SENDER_EMAIL,
            sender_password=SENDER_PASSWORD,
            receiver_email=RECEIVER_EMAIL
        )

        intrusion_cooldown = FPS * 3

    if any_fight and fight_cooldown == 0:
        clip_path = None

        if (now - last_saved_time) > save_gap_seconds:
            clip_path = save_clip(
                rolling_buffer.get(),
                alert_type="fight",
                fps=FPS
            )
            print("Fight clip saved:", clip_path)
            print("Fight alerts:", fight_alerts)
            print("Angry faces:", angry_faces)
            last_saved_time = now

        if (now - last_voice_time) > voice_gap_seconds:
            speak_warning("Warning! Possible fight detected.")
            last_voice_time = now

        send_email_alert(
            subject="ALERT: Possible Fight Detected",
            body=(
                f"Possible fight detected by surveillance system.\n"
                f"Fight alerts: {fight_alerts}\n"
                f"Angry faces detected: {len(angry_faces)}\n"
                f"Saved clip: {clip_path}"
            ),
            sender_email=SENDER_EMAIL,
            sender_password=SENDER_PASSWORD,
            receiver_email=RECEIVER_EMAIL
        )

        fight_cooldown = FPS * 3

    if any_suspicious and suspicious_cooldown == 0:
        clip_path = None

        if (now - last_saved_time) > save_gap_seconds:
            clip_path = save_clip(
                rolling_buffer.get(),
                alert_type="suspicious",
                fps=FPS
            )
            print("Suspicious movement clip saved:", clip_path)
            last_saved_time = now

        if (now - last_voice_time) > voice_gap_seconds:
            speak_warning("Warning! Suspicious movement detected.")
            last_voice_time = now

        send_email_alert(
            subject="ALERT: Suspicious Movement Detected",
            body=(
                f"Suspicious movement detected by surveillance system.\n"
                f"Person IDs: {suspicious_ids}\n"
                f"Saved clip: {clip_path}"
            ),
            sender_email=SENDER_EMAIL,
            sender_password=SENDER_PASSWORD,
            receiver_email=RECEIVER_EMAIL
        )

        suspicious_cooldown = FPS * 3

    cv2.imwrite(LIVE_FRAME_PATH, frame)
    cv2.imshow("Advanced Smart Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
