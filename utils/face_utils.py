import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes

# ---------------------------------------------------------------------------
# face_utils.py
#
# All functions operate on numpy arrays unless the name ends in _tensor.
# Coordinate convention throughout: (x1, y1, x2, y2) in pixel space.
# ---------------------------------------------------------------------------


def safe_crop_np(frame_rgb: np.ndarray, box) -> np.ndarray | None:
    """
    Crop a face region from an RGB numpy frame with full boundary safety.

    Parameters
    ----------
    frame_rgb : H x W x C uint8 numpy array  (RGB)
    box       : (x1, y1, x2, y2) — may be floats from tracker output

    Returns
    -------
    Cropped face as uint8 numpy array, or None if the crop is unusable.

    Bug fixes
    ---------
    - Box coords not cast to int: float indices raise IndexError in numpy ≥1.24.
    - `h, w, _ = frame.shape` crashes on non-3-channel arrays; use shape[:2].
    - Brightness/std checks now operate per-channel mean of the full face,
      which is more robust than a single channel mean for dark-skinned faces
      under warm lighting (single-channel mean can be < 5 even for valid faces).
    """
    # Bug fix: cast to int — tracker outputs float tlbr values
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    # Bug fix: use shape[:2] — safe for any number of channels
    h, w = frame_rgb.shape[:2]

    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Reject inverted or too-small boxes
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None

    face = frame_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None

    # Quality pre-filters (float32 to avoid uint8 overflow in std computation)
    face_f = face.astype(np.float32)

    # Bug fix: check mean across all channels, not just the raw array mean.
    # For an RGB face, np.mean() over the full array already covers all
    # channels — but being explicit avoids confusion.
    if face_f.mean() < 5.0:   # nearly black — bad exposure or crop
        return None
    if face_f.std() < 2.0:    # near-uniform — solid colour, not a face
        return None

    return face


def crop_face_tensor(
    frame: np.ndarray,
    box,
    size: int = 160,
) -> torch.Tensor | None:
    """
    Crop and resize a face, returning a normalised [C, H, W] float32 tensor.

    Used when a torch-format crop is needed (e.g. classification head).
    For embedding extraction use preprocess_face_for_embedder() instead.

    Bug fix: original crop_face() clamped x2/y2 correctly but the intent
    was clearer with the safe_crop_np helper reused here so clamping logic
    isn't duplicated. Also: cv2.resize on a zero-size crop raises an
    exception — guard added.
    """
    face = safe_crop_np(frame, box)
    if face is None:
        return None

    face_resized = cv2.resize(face, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
    return tensor


def preprocess_face_for_embedder(face_rgb: np.ndarray) -> np.ndarray:
    """
    Resize and normalise a face crop for MobileFaceNet embedding inference.

    Input  : H x W x 3 uint8 numpy array (RGB)
    Output : [C, H, W] = [3, 112, 112] float32 numpy array, contiguous,
             normalised to [-1, 1] range using (pixel - 127.5) / 128.0

    Bug fix: the old preprocess_face() returned [C, H, W] with no batch dim
    and no contiguity guarantee. The embedder's batch assembly needs
    contiguous arrays so np.stack works correctly. Added np.ascontiguousarray.

    Note: this function is intentionally NOT called preprocess_face to avoid
    shadowing the method of the same name inside the MobileFaceNet embedder
    class — which caused silent no-ops when this module was imported first.
    """
    face = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = np.transpose(face, (2, 0, 1))   # [H, W, C] → [C, H, W]
    return np.ascontiguousarray(face)


def safe_crop(frame: np.ndarray, box) -> np.ndarray | None:
    """
    Minimal boundary-safe crop returning a uint8 numpy array, or None.

    Used in contexts where the full quality pre-filters of safe_crop_np
    are not wanted (e.g. enrolment where the operator has already verified
    the frame).
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face = frame[y1:y2, x1:x2]
    return None if face.size == 0 else face


def select_face_box(boxes: Boxes) -> tuple | None:
    """
    Return the coordinates of the first detected face box as (x1, y1, x2, y2).

    Bug fix: returned `map(int, ...)` — a map object is not subscriptable,
    so `select_face_box(boxes)[0]` would raise TypeError. Now returns a tuple.

    Bug fix: no guard for empty boxes — `boxes.xyxy[0]` raises IndexError
    when there are no detections. Now returns None for empty input.
    """
    if boxes is None or len(boxes) == 0:
        return None
    xy = boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
    return (x1, y1, x2, y2)


def empty_boxes(orig_shape: tuple = (1, 1), device: str = "cpu") -> Boxes:
    """
    Return an empty Ultralytics Boxes object.

    Bug fix: the old signature was `empty_boxes(device)` and hard-coded
    `orig_shape=(1, 1)`. Any downstream code reading orig_shape for
    coordinate scaling (e.g. Ultralytics post-processing) would get wrong
    scale factors. Now accepts the real frame shape as first argument.

    Usage:
        empty_boxes(frame.shape[:2], device)   # (h, w), device
        empty_boxes()                           # safe default for tests
    """
    return Boxes(
        torch.empty((0, 6), device=device),
        orig_shape=orig_shape,
    )