import torch
import torch.nn.functional as F
from utils.hopfield_layer import HopfieldLayer


def pool_embeddings(buf, device):
    """
    Hopfield-pooling over buffer embeddings.
    buf: Tensor [N, D] or iterable of [D] tensors
    """
    if isinstance(buf, torch.Tensor):
        buf_tensor = buf.to(device, non_blocking=True)
    else:
        buf_tensor = torch.stack(list(buf)).to(device)

    mean_init = buf_tensor.mean(dim=0)
    hop_buf = HopfieldLayer(buf_tensor, device=device)
    pooled = hop_buf.refine(mean_init)
    return pooled


def refine_identity(pooled, hop):
    """
    pooled: [D]
    returns:
        refined: [D]
        E_before: float
        E_after: float
        delta_E: float
    """
    E_before = hop.energy(pooled)
    refined = hop.refine(pooled)
    E_after = hop.energy(refined)

    delta_E = E_before - E_after
    return refined, E_before, E_after, delta_E


def identify_person(
    refined,
    gallery,
    id_names,
    delta,
    threshold=0.7,
    delta_threshold=0.2,
):
    """
    GPU-safe cosine similarity identity matching.
    """

    if delta < delta_threshold:
        return "Unknown", 0.0

    if gallery.numel() == 0:
        return "Unknown", 0.0

    device = refined.device

    # ---- FP32 compute, FP16 storage ----
    gallery_f32 = gallery.to(device, non_blocking=True).float()
    refined_f32 = refined.float()

    scores = torch.matmul(gallery_f32, refined_f32)
    best_score, best_idx = torch.max(scores, dim=0)

    best_score = float(best_score)
    best_name = id_names[int(best_idx)]

    if best_score < threshold:
        return "Unknown", best_score

    return best_name, best_score
