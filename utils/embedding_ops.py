import torch
import torch.nn.functional as F
from utils.hopfield_layer import HopfieldLayer

def normalize_embeddings(embs):
    return F.normalize(embs, dim=1)

def pool_embeddings(buf, device):
    """
    Hopfield-pooling over buffer embeddings
    turns raw temporal embeddings into a stable identity representation
    """
    # Convert deque or other iterables to list for torch.stack
    if not isinstance(buf, (list, tuple)):
        buf = list(buf)
    
    buf_tensor = torch.stack(buf).to(device)
    mean_init = buf_tensor.mean(dim=0)
    hop_buf = HopfieldLayer(buf_tensor, device=device)
    pooled = hop_buf.refine(mean_init)
    return pooled
    
def refine_identity(pooled, hop):
    """
    pooled: [512]
    returns:
        refined: [512]
        E_before: float
        E_after: float
        delta_E: float
    """
    E_before = hop.energy(pooled)
    refined = hop.refine(pooled)
    E_after = hop.energy(refined)

    delta_E = E_before - E_after
    return refined, E_before, E_after, delta_E



import torch

def identify_person(
    refined,
    gallery,
    id_names,
    delta,
    threshold=0.7,
    delta_threshold=0.2,
):
    """
    GPU-safe cosine / dot similarity identity matching.
    Returns (name, score).
    """

    # ---- Delta gate: unstable identity ----
    if delta < delta_threshold:
        return "Unknown", 0.0

    # ---- Empty gallery safety ----
    if gallery.numel() == 0:
        return "Unknown", 0.0

    # ---- Ensure same device (no silent CPU syncs) ----
    device = refined.device
    gallery = gallery.to(device, non_blocking=True)

    # ---- Similarity ----
    scores = torch.matmul(gallery, refined)  # (N,)
    best_score, best_idx = torch.max(scores, dim=0)

    best_score = float(best_score)
    best_name = id_names[int(best_idx)]

    # ---- Similarity threshold gate ----
    if best_score < threshold:
        return "Unknown", best_score

    return best_name, best_score
