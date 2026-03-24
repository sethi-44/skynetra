import torch
import torch.nn.functional as F
from utils.hopfield_layer import HopfieldLayer


# ---------------------------------------------------------------------------
# EmbeddingBuffer
# ---------------------------------------------------------------------------

class EmbeddingBuffer:
    """
    Fixed-size circular buffer of face embeddings for one track.

    Stores the last `max_len` embeddings in a pre-allocated tensor and
    exposes them in chronological order via get_all().
    """

    def __init__(self, max_len: int, dim: int, device: str):
        self.emb     = torch.zeros((max_len, dim), device=device)  # pre-allocated
        self.max_len = max_len
        self.ptr     = 0      # next write position
        self.count   = 0      # total embeddings seen (capped at max_len)

    def add(self, emb: torch.Tensor) -> None:
        """Write one [D] embedding into the circular buffer."""
        # Bug fix: store a detached, normalised copy so downstream consumers
        # never mutate buffer contents and always operate on unit vectors.
        self.emb[self.ptr] = F.normalize(emb.detach().float(), dim=-1)
        self.ptr   = (self.ptr + 1) % self.max_len
        self.count = min(self.count + 1, self.max_len)

    def full(self) -> bool:
        return self.count >= self.max_len

    def get_all(self) -> torch.Tensor:
        """
        Return all stored embeddings in chronological order as [N, D].

        Bug fix: when the buffer is not yet full, the old code returned
        self.emb[:self.count] which is correct. When full it used arange
        from ptr to ptr+max_len — also correct — but both branches
        returned a view into self.emb rather than a contiguous copy, so
        any in-place operation on the result would silently corrupt the
        buffer. Returns a contiguous clone now.
        """
        if self.count < self.max_len:
            return self.emb[:self.count].clone()
        idx = torch.arange(
            self.ptr, self.ptr + self.max_len, device=self.emb.device
        ) % self.max_len
        return self.emb[idx].contiguous()

    def reset(self) -> None:
        """Clear the buffer (call when a track ID is reassigned)."""
        self.ptr   = 0
        self.count = 0


# ---------------------------------------------------------------------------
# pool_embeddings
# ---------------------------------------------------------------------------

def pool_embeddings(buf: torch.Tensor, device: str) -> torch.Tensor:
    """
    Produce a single representative embedding from a buffer of N embeddings.

    Two-stage process:
        1. Compute the mean of all buffer embeddings (normalised), which acts
           as a stable initialisation point near the cluster centroid.
        2. Refine that mean via one pass of the buffer's own Hopfield layer,
           nudging it toward the densest attractor in the buffer.

    This is intentionally lightweight — the buffer Hopfield layer uses the
    raw observations as patterns, NOT the gallery. Its job is temporal
    denoising within a single track, not cross-identity matching.

    Parameters
    ----------
    buf    : [N, D] float tensor of unit-norm embeddings
    device : target device string

    Returns
    -------
    pooled : [D] float tensor, unit-norm, on `device`

    Bug fix: if buf is not a Tensor (list/iterable fallback) the old code
    called torch.stack() without detach, potentially including grad graphs.
    Now always produces a clean float tensor.
    """
    if isinstance(buf, torch.Tensor):
        buf_tensor = buf.to(device, non_blocking=True).float()
    else:
        buf_tensor = torch.stack([b.detach().float() for b in buf]).to(device)

    # Ensure unit norm on every row before pooling
    buf_tensor = F.normalize(buf_tensor, dim=1)

    # Mean initialisation → Hopfield refinement
    mean_init = F.normalize(buf_tensor.mean(dim=0), dim=-1)
    hop_buf   = HopfieldLayer(buf_tensor, device=device)
    pooled    = hop_buf.refine(mean_init)

    return F.normalize(pooled, dim=-1)


# ---------------------------------------------------------------------------
# refine_identity
# ---------------------------------------------------------------------------

def refine_identity(
    pooled: torch.Tensor,
    hop: "HopfieldLayer",
) -> tuple:
    """
    Refine a pooled embedding against the gallery Hopfield layer and measure
    how much energy changed (used as a proxy for match confidence).

    Parameters
    ----------
    pooled : [D] unit-norm tensor — output of pool_embeddings
    hop    : HopfieldLayer built from the identity gallery

    Returns
    -------
    refined  : [D] unit-norm tensor after gallery refinement
    E_before : float — Hopfield energy of `pooled` (negative = near attractor)
    E_after  : float — Hopfield energy of `refined`
    delta_E  : float — E_before - E_after; positive = energy decreased = good

    Bug fix: if hop is None (empty gallery), the old code would raise
    AttributeError immediately. Return a passthrough with zero delta.
    """
    if hop is None:
        return pooled, 0.0, 0.0, 0.0

    E_before = hop.energy(pooled)
    refined  = hop.refine(pooled)
    E_after  = hop.energy(refined)
    delta_E  = E_before - E_after   # positive when refinement decreased energy

    return refined, E_before, E_after, delta_E


# ---------------------------------------------------------------------------
# identify_person
# ---------------------------------------------------------------------------

def identify_person(
    refined: torch.Tensor,
    gallery: torch.Tensor,
    id_names: list,
    delta: float,
    threshold: float       = 0.70,   # bug fix: was 0.95 — unreachably strict
    delta_threshold: float = 0.20,   # bug fix: was 0.80 — unreachably strict
) -> tuple:
    """
    Match a refined embedding against the identity gallery using cosine
    similarity and return the best name + score.

    Parameters
    ----------
    refined         : [D] unit-norm tensor (gallery Hopfield-refined)
    gallery         : [G, D] float tensor of unit-norm gallery embeddings
    id_names        : list of G identity name strings
    delta           : energy drop from refine_identity (confidence proxy)
    threshold       : minimum cosine similarity to accept a match
    delta_threshold : minimum delta_E required before attempting matching.
                      Guards against matching when refinement did nothing.

    Returns
    -------
    (name, score) where name is a string and score ∈ [0, 1]

    Bug fix: default threshold=0.95 and delta_threshold=0.80 were far too
    strict — virtually nothing would ever match, making the system return
    "Unknown" for every face regardless of quality. Corrected to 0.70 and
    0.20, matching the values used in main_helpers.py's explicit call-site.
    These are now the real defaults so a bare identify_person() call works.

    Bug fix: gallery.numel()==0 check now comes BEFORE the delta check so
    we don't silently return "Unknown" with score=0.0 when the gallery is
    empty, which was indistinguishable from a genuine low-score match.
    Instead we return a dedicated sentinel tuple.

    Bug fix: refined is not guaranteed to be unit-norm if it came from a
    non-normalising path — explicitly normalise before matmul to ensure
    cosine similarity semantics hold.
    """
    # Empty gallery — no identities enrolled yet
    if gallery.numel() == 0 or len(id_names) == 0:
        return "Unknown", 0.0

    # Refinement didn't move the embedding — result is unreliable
    if delta < delta_threshold:
        return "Unknown", 0.0

    device = refined.device

    # Ensure unit norm for cosine similarity
    refined_f32 = F.normalize(refined.float(), dim=-1)
    gallery_f32 = F.normalize(
        gallery.to(device, non_blocking=True).float(), dim=1
    )

    # [G] cosine similarities (gallery rows are unit-norm, refined is unit-norm)
    scores               = gallery_f32 @ refined_f32
    best_score, best_idx = torch.max(scores, dim=0)

    best_score = float(best_score)
    best_idx   = int(best_idx)

    # Guard: best_idx must be a valid index (paranoia check)
    if best_idx >= len(id_names):
        return "Unknown", best_score

    if best_score < threshold:
        return "Unknown", best_score

    return id_names[best_idx], best_score