from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30)

def update_tracks(detections, frame):
    formatted = []
    for (x1, y1, x2, y2), conf in detections:
        formatted.append(([x1, y1, x2 - x1, y2 - y1], conf, None))  # xywh
    tracks = tracker.update_tracks(formatted, frame=frame)

    tracked_faces = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        tracked_faces.append((track_id, (int(l), int(t), int(r), int(b))))
    return tracked_faces
