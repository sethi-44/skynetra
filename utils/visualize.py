import cv2
import numpy as np

# ---------------------------------------------------------------------------
# SkyNetra overlay renderer
#
# Design language
# ---------------
#   - Corner-accent bounding box (4 L-shaped corners, no full rectangle)
#   - Frosted semi-transparent info card with a coloured header strip
#   - Confidence bar under the card (proportional fill)
#   - Two-tone text: bright primary line, muted secondary lines
#   - Card flips below the face when there is no room above
#   - All text is pure ASCII (cv2.putText has no Unicode support)
# ---------------------------------------------------------------------------

# ── Palette (BGR) ────────────────────────────────────────────────────────────
_C = {
    "tracking":  (0,   220,   0),   # green  — no identity yet
    "confident": (255, 150,   0),   # blue   — identity locked
    "low_rel":   (0,   220, 220),   # yellow — low reliability
    "abstain":   (50,   50, 232),   # red    — do not trust
    "rejected":  (0,   165, 255),   # orange — bad input quality
}

# Muted (dimmed) variant for secondary text lines
_DIM = {k: tuple(int(c * 0.60) for c in v) for k, v in _C.items()}

# ── Layout constants ──────────────────────────────────────────────────────────
_FONT         = cv2.FONT_HERSHEY_SIMPLEX
_SCALE_PRI    = 0.52   # primary line (name / state label)
_SCALE_SEC    = 0.42   # secondary lines (scores, warning)
_TH_PRI       = 1
_TH_SEC       = 1
_LINE_H_PRI   = 19     # gap from primary baseline to first secondary baseline
_LINE_H_SEC   = 15     # gap between secondary baselines
_CARD_PAD_X   = 7      # horizontal inner padding
_CARD_PAD_Y   = 5      # vertical inner padding
_HEADER_H     = 18     # coloured header strip height in px
_CORNER_LEN   = 12     # length of each corner accent arm
_CORNER_TH    = 3      # thickness of corner accent lines
_BAR_H        = 4      # confidence bar height
_BAR_GAP      = 3      # gap between card bottom and confidence bar
_CARD_ALPHA   = 0.55   # frosted card opacity (0=transparent, 1=solid)
_HEADER_ALPHA = 0.82   # header strip opacity


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _corner_box(img, x1, y1, x2, y2, color):
    """Four L-shaped corner accents instead of a full bounding rectangle."""
    L = _CORNER_LEN
    segs = [
        ((x1, y1 + L), (x1, y1), (x1 + L, y1)),
        ((x2 - L, y1), (x2, y1), (x2, y1 + L)),
        ((x1, y2 - L), (x1, y2), (x1 + L, y2)),
        ((x2 - L, y2), (x2, y2), (x2, y2 - L)),
    ]
    for a, b, c in segs:
        cv2.line(img, a, b, color, _CORNER_TH, cv2.LINE_AA)
        cv2.line(img, b, c, color, _CORNER_TH, cv2.LINE_AA)


def _frosted_rect(img, x1, y1, x2, y2, color, alpha):
    """Semi-transparent filled rectangle blended with a colour tint."""
    fh, fw = img.shape[:2]
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(fw, x2), min(fh, y2)
    if rx2 <= rx1 or ry2 <= ry1:
        return
    roi     = img[ry1:ry2, rx1:rx2]
    dark    = np.full_like(roi, 18)
    tinted  = np.full_like(roi, color, dtype=np.uint8)
    blended = cv2.addWeighted(dark, 0.55, tinted, 0.45, 0)
    img[ry1:ry2, rx1:rx2] = cv2.addWeighted(roi, 1.0 - alpha, blended, alpha, 0)


def _text_w(text, scale, thickness):
    (w, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    return w


def _put(img, text, x, y, scale, color, thickness):
    """Text with a 1-px dark drop-shadow for readability on any background."""
    cv2.putText(img, text, (x + 1, y + 1), _FONT, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def _conf_bar(img, x1, y, width, conf, color):
    """Horizontal confidence bar: dark track + coloured fill + border."""
    fh, fw = img.shape[:2]
    bx1, by1 = max(0, x1), max(0, y)
    bx2, by2 = min(fw, x1 + width), min(fh, y + _BAR_H)
    if bx2 <= bx1 or by2 <= by1:
        return
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (35, 35, 35), -1)
    fill = bx1 + int((bx2 - bx1) * float(np.clip(conf, 0.0, 1.0)))
    if fill > bx1:
        cv2.rectangle(img, (bx1, by1), (fill, by2), color, -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), tuple(int(c * 0.7) for c in color), 1)


# ---------------------------------------------------------------------------
# Card content builder
# ---------------------------------------------------------------------------

def _build_card(info, tid):
    """
    Return (primary_line, secondary_lines, conf_float, state_key).
    All strings are pure ASCII.
    """
    if info is None:
        return f"ID {tid}", [], 0.0, "tracking"

    name  = info.get("name", "Unknown")
    conf  = float(info.get("id_conf", 0.0))

    # REJECTED ---------------------------------------------------------------
    if name == "REJECTED":
        sec = []
        reasons = info.get("quality_reasons", [])
        if reasons:
            sec.append(reasons[0])
        q = info.get("quality")
        if q is not None:
            sec.append(f"Quality {q:.2f}")
        return "REJECTED", sec, 0.0, "rejected"

    # ABSTAIN ----------------------------------------------------------------
    if info.get("is_abstain", False):
        sec = []
        warning = info.get("warning", "")
        if warning:
            sec.append(f">> {warning}")
        q  = info.get("quality")
        ts = info.get("temporal_score")
        if q is not None and ts is not None:
            sec.append(f"Q {q:.2f}  T {ts:.2f}")
        return "!! ABSTAIN", sec, conf, "abstain"

    # LOW RELIABILITY / CONFIDENT --------------------------------------------
    state   = "low_rel" if info.get("quality_level") == "LOW" else "confident"
    primary = f"{name}  {conf:.2f}"
    sec     = []

    # Quality + temporal on one compact line
    q  = info.get("quality")
    ts = info.get("temporal_score")
    if q is not None and ts is not None:
        ql = (info.get("quality_level")  or "?")[0]   # H / M / L
        tl = (info.get("temporal_level") or "?")[0]
        sec.append(f"Q {q:.2f}({ql})  T {ts:.2f}({tl})")
    elif q is not None:
        sec.append(f"Q {q:.2f} ({info.get('quality_level', '?')})")

    # Hopfield refinement
    dE = info.get("dE")
    if dE is not None:
        r = "HIGH" if dE > 0.2 else "LOW"
        sec.append(f"Refine {r}  dE {dE:.3f}")

    # Warning (highest priority secondary — must not be truncated)
    warning = info.get("warning", "")
    if warning:
        sec.insert(0, f">> {warning}")   # prepend so it survives the cap

    return primary, sec, conf, state


# ---------------------------------------------------------------------------
# Card geometry
# ---------------------------------------------------------------------------

def _card_dims(primary, secondary, face_w):
    """Return (card_width, card_height) in pixels."""
    widths  = [_text_w(primary, _SCALE_PRI, _TH_PRI)]
    widths += [_text_w(s, _SCALE_SEC, _TH_SEC) for s in secondary]
    cw = max(max(widths) + _CARD_PAD_X * 2, face_w)
    ch = (
        _HEADER_H
        + _CARD_PAD_Y
        + _LINE_H_PRI
        + min(len(secondary), 4) * _LINE_H_SEC
        + _CARD_PAD_Y
    )
    return cw, ch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_tracks(frame: np.ndarray, tracks, track_info: dict) -> np.ndarray:
    """
    Draw per-track overlays on a BGR frame.

    For each active track renders:
      - Corner-accent bounding box
      - Frosted info card with coloured header strip
      - Confidence bar beneath the card

    Parameters
    ----------
    frame      : H x W x 3 BGR uint8 numpy array (modified in-place)
    tracks     : list of ByteTrack STrack objects
    track_info : dict  track_id -> info dict built by main_helpers.py

    Returns
    -------
    The same frame with annotations drawn.
    """
    fh, fw = frame.shape[:2]

    for t in tracks:
        if not t.is_activated:
            continue

        tid             = t.track_id
        x1, y1, x2, y2 = map(int, t.tlbr)
        info            = track_info.get(tid)

        primary, secondary, conf, state = _build_card(info, tid)
        color = _C[state]
        dim   = _DIM[state]

        # ── 1. Corner-accent bounding box ────────────────────────────────────
        _corner_box(frame, x1, y1, x2, y2, color)

        # ── 2. Card geometry ─────────────────────────────────────────────────
        face_w    = x2 - x1
        cw, ch    = _card_dims(primary, secondary, face_w)
        bar_total = _BAR_H + _BAR_GAP

        # Align left edge with face box, clamp to frame width
        cx1 = max(0, min(x1, fw - cw))
        cx2 = cx1 + cw

        # Place above if there's room; fall back to below
        if y1 - ch - bar_total - 2 >= 0:
            cy2 = y1 - 2
            cy1 = cy2 - ch
            bar_y = cy2 + _BAR_GAP
        else:
            cy1 = y2 + 2
            cy2 = cy1 + ch
            bar_y = cy2 + _BAR_GAP

        # ── 3. Frosted card body ─────────────────────────────────────────────
        _frosted_rect(frame, cx1, cy1, cx2, cy2, color, _CARD_ALPHA)

        # Coloured header strip
        _frosted_rect(frame, cx1, cy1, cx2, cy1 + _HEADER_H, color, _HEADER_ALPHA)

        # Crisp top border
        cv2.line(frame, (cx1, cy1), (cx2, cy1), color, 1, cv2.LINE_AA)

        # ── 4. Text ──────────────────────────────────────────────────────────
        # Primary line — sits in the header strip
        ty_pri = cy1 + _HEADER_H - 4
        _put(frame, primary,
             cx1 + _CARD_PAD_X, ty_pri,
             _SCALE_PRI, (255, 255, 255), _TH_PRI)

        # Secondary lines — sit in the body below the header
        ty0 = cy1 + _HEADER_H + _CARD_PAD_Y + _LINE_H_SEC - 2
        for i, line in enumerate(secondary[:4]):
            _put(frame, line,
                 cx1 + _CARD_PAD_X, ty0 + i * _LINE_H_SEC,
                 _SCALE_SEC, dim, _TH_SEC)

        # ── 5. Confidence bar ────────────────────────────────────────────────
        if state not in ("rejected",) and conf > 0.0:
            _conf_bar(frame, cx1, bar_y, cw, conf, color)

    return frame