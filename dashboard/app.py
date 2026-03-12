from flask import Flask, render_template_string, send_from_directory
import os
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS_FOLDER = os.path.join(BASE_DIR, "clips")
FRAME_FOLDER = os.path.join(BASE_DIR, "dashboard_frames")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Surveillance Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #111;
            color: white;
            padding: 20px;
        }
        h1 {
            color: #00d4ff;
        }
        h2 {
            margin-top: 30px;
        }
        .live-feed {
            background: #1e1e1e;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 25px;
        }
        .live-feed img {
            width: 100%;
            max-width: 800px;
            border-radius: 8px;
            background: black;
        }
        .card {
            background: #1e1e1e;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .intrusion {
            color: orange;
            font-weight: bold;
        }
        .fight {
            color: red;
            font-weight: bold;
        }
        .suspicious {
            color: magenta;
            font-weight: bold;
        }
        .unknown {
            color: lightgray;
            font-weight: bold;
        }
        video {
            width: 100%;
            max-width: 700px;
            margin-top: 10px;
            border-radius: 8px;
            background: black;
        }
        .meta {
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <h1>AI Smart Surveillance Dashboard</h1>
    <p>Auto-refreshes every 5 seconds</p>

    
    
</div>

    <h2>Recent Alerts</h2>
    {% if alerts %}
        {% for alert in alerts %}
            <div class="card">
                <div class="meta"><strong>Time:</strong> {{ alert.time }}</div>
                <div class="meta"><strong>Type:</strong>
                    <span class="{{ alert.type }}">{{ alert.type }}</span>
                </div>
                <div class="meta"><strong>File:</strong> {{ alert.file }}</div>

                <video controls>
                    <source src="/clips/{{ alert.file }}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
        {% endfor %}
    {% else %}
        <p>No alerts found yet.</p>
    {% endif %}
</body>
</html>
"""


def get_alerts():
    alerts = []

    if not os.path.exists(CLIPS_FOLDER):
        return alerts

    files = sorted(
        os.listdir(CLIPS_FOLDER),
        key=lambda f: os.path.getmtime(os.path.join(CLIPS_FOLDER, f)),
        reverse=True
    )

    for file in files:
        if not file.endswith(".mp4"):
            continue

        full_path = os.path.join(CLIPS_FOLDER, file)
        modified_time = datetime.fromtimestamp(
            os.path.getmtime(full_path)
        ).strftime("%Y-%m-%d %H:%M:%S")

        if file.startswith("intrusion"):
            alert_type = "intrusion"
        elif file.startswith("fight"):
            alert_type = "fight"
        elif file.startswith("suspicious"):
            alert_type = "suspicious"
        else:
            alert_type = "unknown"

        alerts.append({
            "time": modified_time,
            "type": alert_type,
            "file": file
        })

    return alerts


@app.route("/")
def dashboard():
    alerts = get_alerts()
    live_frame_exists = os.path.exists(os.path.join(FRAME_FOLDER, "live.jpg"))

    return render_template_string(
        HTML,
        alerts=alerts,
        live_frame_exists=live_frame_exists,
        timestamp=int(datetime.now().timestamp())
    )


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    return send_from_directory(CLIPS_FOLDER, filename)


@app.route("/live_frame/<path:filename>")
def serve_live_frame(filename):
    return send_from_directory(FRAME_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)