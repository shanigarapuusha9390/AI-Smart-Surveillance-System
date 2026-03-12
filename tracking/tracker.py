from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:

    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, frame):

        tracks = self.tracker.update_tracks(
            detections,
            frame=frame
        )

        persons = []

        for track in tracks:

            if not track.is_confirmed():
                continue

            l, t, r, b = track.to_ltrb()

            persons.append({
                "id": track.track_id,
                "bbox": [l, t, r, b]
            })

        return persons