import math
from collections import defaultdict, deque


class FightDetector:
    """
    Advanced fight detector:
    - Requires TWO people
    - Uses pose keypoints (wrists, elbows, shoulders, hips)
    - Uses proximity + arm motion + body motion
    - Can use angry face support signal
    - Requires persistence across multiple frames
    """

    # COCO keypoint indexes
    NOSE = 0
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12

    def __init__(self):
        self.track_history = defaultdict(lambda: deque(maxlen=12))
        self.pair_counter = defaultdict(int)

        # Tunable thresholds
        self.close_distance_ratio = 2.0
        self.arm_speed_ratio = 0.22
        self.body_speed_ratio = 0.10
        self.contact_distance_ratio = 0.9
        self.min_persistent_frames = 6

        # Score thresholds
        self.base_fight_threshold = 0.62
        self.final_fight_threshold = 0.72

        # Angry face support
        self.face_support_bonus = 0.18
        self.face_match_distance_ratio = 1.2

    def update_track_pose(self, track_id, bbox, keypoints, kp_conf=None):
        center = self._bbox_center(bbox)
        scale = self._body_scale(keypoints)

        if scale <= 1:
            return

        entry = {
            "bbox": bbox,
            "center": center,
            "scale": scale,
            "keypoints": keypoints,
            "kp_conf": kp_conf if kp_conf is not None else [1.0] * len(keypoints),
        }
        self.track_history[track_id].append(entry)

    def detect(self, persons_with_pose, angry_faces=None):
        """
        persons_with_pose format:
        [
          {
            "id": 1,
            "bbox": [...],
            "keypoints": [...],
            "kp_conf": [...]
          },
          ...
        ]

        angry_faces format:
        [
          {
            "bbox": (x1, y1, x2, y2),
            "score": 0.82
          },
          ...
        ]

        Returns:
        [
          {
            "pair": (1, 2),
            "score": 0.84,
            "base_score": 0.68,
            "face_score": 0.16
          },
          ...
        ]
        """
        if angry_faces is None:
            angry_faces = []

        alerts = []
        active_pairs = set()

        valid_people = []
        for p in persons_with_pose:
            pid = p["id"]
            kps = p.get("keypoints")

            if kps is None:
                continue

            self.update_track_pose(pid, p["bbox"], kps, p.get("kp_conf"))
            features = self._person_motion_features(pid)
            if features is None:
                continue

            valid_people.append({
                "id": pid,
                "bbox": p["bbox"],
                "keypoints": kps,
                "features": features
            })

        for i in range(len(valid_people)):
            for j in range(i + 1, len(valid_people)):
                a = valid_people[i]
                b = valid_people[j]

                pair_key = tuple(sorted((a["id"], b["id"])))

                base_score = self._pair_base_fight_score(a, b)
                face_score = self._pair_face_support_score(a, b, angry_faces)
                final_score = self._clamp01(base_score + face_score)

                # Require body/pose evidence first, then face boosts it
                if base_score >= self.base_fight_threshold and final_score >= self.final_fight_threshold:
                    self.pair_counter[pair_key] += 1
                    active_pairs.add(pair_key)

                    if self.pair_counter[pair_key] >= self.min_persistent_frames:
                        alerts.append({
                            "pair": pair_key,
                            "score": round(final_score, 3),
                            "base_score": round(base_score, 3),
                            "face_score": round(face_score, 3)
                        })
                else:
                    self.pair_counter[pair_key] = max(0, self.pair_counter[pair_key] - 1)

        for pair_key in list(self.pair_counter.keys()):
            if pair_key not in active_pairs:
                self.pair_counter[pair_key] = max(0, self.pair_counter[pair_key] - 1)

        return alerts

    def _pair_base_fight_score(self, a, b):
        fa = a["features"]
        fb = b["features"]

        avg_scale = max(1.0, (fa["scale"] + fb["scale"]) / 2.0)
        center_dist = self._dist(fa["center"], fb["center"]) / avg_scale
        close_score = self._clamp01(1.0 - (center_dist / self.close_distance_ratio))

        arm_score = self._clamp01(
            (fa["arm_speed"] + fb["arm_speed"]) / (2 * self.arm_speed_ratio)
        )
        body_score = self._clamp01(
            (fa["body_speed"] + fb["body_speed"]) / (2 * self.body_speed_ratio)
        )

        contact_score = max(
            self._arm_to_torso_contact_score(a["keypoints"], b["keypoints"], avg_scale),
            self._arm_to_torso_contact_score(b["keypoints"], a["keypoints"], avg_scale),
        )

        score = (
            0.30 * close_score +
            0.30 * arm_score +
            0.15 * body_score +
            0.25 * contact_score
        )
        return score

    def _pair_face_support_score(self, a, b, angry_faces):
        if not angry_faces:
            return 0.0

        fa = a["features"]
        fb = b["features"]
        avg_scale = max(1.0, (fa["scale"] + fb["scale"]) / 2.0)

        score_a = self._person_angry_face_score(a, angry_faces, avg_scale)
        score_b = self._person_angry_face_score(b, angry_faces, avg_scale)

        # stronger support if both participants seem angry
        if score_a > 0 and score_b > 0:
            combined = (score_a + score_b) / 2.0
        else:
            combined = max(score_a, score_b) * 0.7

        return self.face_support_bonus * self._clamp01(combined)

    def _person_angry_face_score(self, person, angry_faces, scale):
        px, py = self._bbox_center(person["bbox"])
        best = 0.0

        for face in angry_faces:
            fx1, fy1, fx2, fy2 = face["bbox"]
            fcx = (fx1 + fx2) / 2.0
            fcy = (fy1 + fy2) / 2.0

            d = self._dist((px, py), (fcx, fcy)) / max(scale, 1.0)
            proximity = self._clamp01(1.0 - (d / self.face_match_distance_ratio))

            emotion_score = float(face.get("score", 0.0))
            candidate = proximity * emotion_score
            best = max(best, candidate)

        return best

    def _person_motion_features(self, track_id):
        hist = self.track_history[track_id]
        if len(hist) < 4:
            return None

        last = hist[-1]
        prev = hist[-2]

        center = last["center"]
        scale = last["scale"]

        body_speed = self._dist(last["center"], prev["center"]) / max(scale, 1.0)

        arm_points_last = self._arm_points(last["keypoints"])
        arm_points_prev = self._arm_points(prev["keypoints"])

        arm_speeds = []
        for p_now, p_prev in zip(arm_points_last, arm_points_prev):
            if p_now is not None and p_prev is not None:
                arm_speeds.append(self._dist(p_now, p_prev) / max(scale, 1.0))

        arm_speed = sum(arm_speeds) / len(arm_speeds) if arm_speeds else 0.0

        return {
            "center": center,
            "scale": scale,
            "body_speed": body_speed,
            "arm_speed": arm_speed,
        }

    def _arm_to_torso_contact_score(self, attacker_kps, victim_kps, scale):
        wrists = [
            self._safe_point(attacker_kps, self.L_WRIST),
            self._safe_point(attacker_kps, self.R_WRIST),
        ]

        torso_points = [
            self._safe_point(victim_kps, self.L_SHOULDER),
            self._safe_point(victim_kps, self.R_SHOULDER),
            self._safe_point(victim_kps, self.L_HIP),
            self._safe_point(victim_kps, self.R_HIP),
        ]
        torso_points = [p for p in torso_points if p is not None]

        if not torso_points:
            return 0.0

        torso_center = (
            sum(p[0] for p in torso_points) / len(torso_points),
            sum(p[1] for p in torso_points) / len(torso_points)
        )

        best = 0.0
        for w in wrists:
            if w is None:
                continue
            d = self._dist(w, torso_center) / max(scale, 1.0)
            score = self._clamp01(1.0 - (d / self.contact_distance_ratio))
            best = max(best, score)

        return best

    def _arm_points(self, kps):
        return [
            self._safe_point(kps, self.L_ELBOW),
            self._safe_point(kps, self.R_ELBOW),
            self._safe_point(kps, self.L_WRIST),
            self._safe_point(kps, self.R_WRIST),
        ]

    def _body_scale(self, kps):
        ls = self._safe_point(kps, self.L_SHOULDER)
        rs = self._safe_point(kps, self.R_SHOULDER)
        lh = self._safe_point(kps, self.L_HIP)
        rh = self._safe_point(kps, self.R_HIP)

        pts = [p for p in [ls, rs, lh, rh] if p is not None]
        if len(pts) < 2:
            return 0.0

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _safe_point(self, kps, idx):
        if kps is None or idx >= len(kps):
            return None
        x, y = kps[idx]
        if x <= 0 and y <= 0:
            return None
        return (float(x), float(y))

    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _clamp01(self, x):
        return max(0.0, min(1.0, x))