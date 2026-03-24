import torch
import torch.nn.functional as F
import math


class HopfieldLayer:
    """
    Modern continuous Hopfield network for embedding retrieval and refinement.

    Energy convention (important — read before touching sign logic):
        E(q) = -log Σ_i exp(β · <q, ξ_i>)

    A query AT an attractor (high similarity to a stored pattern) produces
    large scores → large logsumexp → E is a large negative number.
    A query FAR from all patterns produces small scores → E is closer to 0
    or slightly negative.

    So: lower (more negative) E = more confident = better refinement.
    delta = E_before - E_after > 0  means energy DECREASED = refinement worked.

    The sampler receives (energy_after, delta) and uses:
        raw = 1 - exp(-energy_after / norm)
    Because energy_after is negative for good matches, this produces values
    < 0.5 for confident identities and near 1.0 for poor ones — correct
    uncertainty semantics, but only due to the negative sign. This is now
    documented explicitly so future edits don't silently break it.

    Beta guidance:
        For unit-norm embeddings in D dimensions, cosine similarities cluster
        near 0. At beta=1 the softmax becomes near-uniform (averaging all
        patterns). Use beta = sqrt(D) as a starting point:
            D=256  → beta ≈ 16
            D=512  → beta ≈ 22
        Too high a beta makes the network brittle (winner-takes-all with no
        interpolation). Tune empirically; 8–20 is a reasonable range for faces.
    """

    def __init__(
        self,
        stored_patterns,
        beta: float = None,   # None → auto-set to sqrt(D)
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        stored_patterns : [N, D] Tensor, list of [D] Tensors, or array-like
        beta            : softmax inverse temperature. None = sqrt(embedding_dim).
        device          : "cpu" or "cuda"
        """
        self.device = device

        # ── Parse stored_patterns into a [N, D] float tensor ────────────────
        if isinstance(stored_patterns, (list, tuple)):
            if len(stored_patterns) == 0:
                raise ValueError("stored_patterns must contain at least one pattern")
            stored = (
                torch.stack(stored_patterns)
                if isinstance(stored_patterns[0], torch.Tensor)
                else torch.tensor(stored_patterns, dtype=torch.float32)
            )
        elif isinstance(stored_patterns, torch.Tensor):
            stored = stored_patterns
        else:
            stored = torch.tensor(stored_patterns, dtype=torch.float32)

        if stored.dim() == 1:
            stored = stored.unsqueeze(0)   # [1, D]

        stored = stored.float().to(device)
        self.stored = F.normalize(stored, dim=1)   # unit-norm patterns
        self.n_patterns, self.dim = self.stored.shape

        # Bug fix: beta=1.0 is too low for high-dimensional embeddings —
        # softmax becomes near-uniform, turning retrieval into averaging.
        # Auto-set to sqrt(D) which keeps attention scores in a useful range.
        self.beta = beta if beta is not None else math.sqrt(self.dim)

    # ── Core operations ──────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, query: torch.Tensor) -> torch.Tensor:
        """
        One Hopfield update step: softmax-weighted sum of stored patterns.

        Parameters
        ----------
        query : [D] tensor (any device, any norm)

        Returns
        -------
        retrieved : [D] tensor on self.device, unit-norm
        """
        # Bug fix: move to device BEFORE normalize to avoid redundant copies
        # when query is already on the correct device.
        q = F.normalize(query.to(self.device), dim=-1)   # [D]

        scores    = self.beta * (q @ self.stored.T)       # [N]
        weights   = F.softmax(scores, dim=0)              # [N]
        retrieved = weights @ self.stored                  # [D]
        return F.normalize(retrieved, dim=-1)

    @torch.no_grad()
    def energy(self, query: torch.Tensor) -> float:
        """
        Modern Hopfield energy: E(q) = -log Σ_i exp(β · <q, ξ_i>)

        Lower (more negative) = query is near a stored attractor = confident.
        Higher (closer to 0)  = query is far from all patterns = uncertain.

        For a gallery of N=1, energy = -β · <q, ξ₁> which is a valid
        similarity score but delta will be small — the sampler handles this
        via the refinement_weakness term.

        Returns
        -------
        float  (negative for well-matched queries)
        """
        q      = F.normalize(query.to(self.device), dim=-1)
        scores = self.beta * (q @ self.stored.T)           # [N]
        return (-torch.logsumexp(scores, dim=0)).item()

    @torch.no_grad()
    def refine(
        self,
        query: torch.Tensor,
        max_steps: int = 10,
        alpha: float   = 0.7,
        tol: float     = 1e-4,
    ) -> torch.Tensor:
        """
        Iterative Hopfield refinement: nudge the query toward the nearest
        attractor using a residual blend.

        Parameters
        ----------
        query     : [D] tensor
        max_steps : maximum iterations (increased from 5 → 10 for better
                    convergence on high-beta networks)
        alpha     : residual weight on the current query (momentum).
                    Higher = slower convergence, more stable.
        tol       : energy non-improvement threshold for early stopping.

        Returns
        -------
        q : [D] refined unit-norm tensor on self.device

        Bug fix: the old code broke when `delta < tol` and returned `q`
        (the step BEFORE q_next). Now we assign q = q_next before checking
        convergence, so the returned value is always the most-refined step.

        Bug fix: convergence now tracks energy decrease rather than
        step-to-step cosine distance. Energy non-improvement is the correct
        stopping criterion — the old cosine check could terminate early in
        flat regions of the landscape where consecutive steps look similar
        but the query hasn't reached a proper attractor yet.
        """
        q      = F.normalize(query.to(self.device), dim=-1)
        E_prev = self.energy(q)

        for _ in range(max_steps):
            retrieved = self.update(q)
            q = F.normalize(alpha * q + (1.0 - alpha) * retrieved, dim=-1)

            # Bug fix: check energy improvement, not step similarity.
            # Assign q first so early-exit returns the freshest refined state.
            E_curr = self.energy(q)
            if abs(E_prev - E_curr) < tol:
                break
            E_prev = E_curr

        return q