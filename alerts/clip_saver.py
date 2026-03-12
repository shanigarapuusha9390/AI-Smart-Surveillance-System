import cv2
import os
from datetime import datetime


def save_clip(frames, alert_type="alert", fps=30):
    if not frames:
        return None

    os.makedirs("clips", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clips/{alert_type}_{timestamp}.mp4"

    h, w, _ = frames[0].shape

    writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    for frame in frames:
        writer.write(frame)

    writer.release()
    print(f"Saved clip: {filename}")
    return filename