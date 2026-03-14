# import cv2

# def draw_tracks(frame, tracks, identity_labels):
#     for t in tracks:
#         x1, y1, x2, y2 = map(int, t.tlbr)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#         label = identity_labels.get(t.track_id, f"ID {t.track_id}")
#         cv2.putText(frame, label, (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#     return frame
import cv2

def draw_tracks(frame, tracks, track_info):
    for t in tracks:
        if not t.is_activated:
            continue

        tid = t.track_id
        x1, y1, x2, y2 = map(int, t.tlbr)

        info = track_info.get(tid)

        # ---------------------------
        # Color logic
        # ---------------------------
        if info is None:
            color = (0, 255, 0)   # green → tracking only
        else:
            color = (0, 0, 255)   # red → identity locked

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ---------------------------
        # Text lines
        # ---------------------------
        lines = []

        if info is None:
            lines.append(f"ID {tid}")
        else:
            # primary identity line
            lines.append(f"{info['name']} | id {info['id_conf']:.2f}")

            # telemetry
            if info.get("det_conf") is not None:
                lines.append(f"det {info['det_conf']:.2f}")

            lines.append(f"trk {info['track_conf']:.2f}")
            lines.append(f"E {info['E_after']:.3f}")
            lines.append(f"dE {info['dE']:.3f}")

        # ---------------------------
        # Render stacked text
        # ---------------------------
        ty = max(15, y1 - 6)
        for i, txt in enumerate(lines):
            cv2.putText(
                frame,
                txt,
                (x1, ty + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    return frame
