import cv2

def draw_tracks(frame, tracks, identity_labels):
    for t in tracks:
        x1, y1, x2, y2 = map(int, t[0:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        label = identity_labels.get(t[4], f"ID {t[4]}")
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame
