import torch
import torch.nn.functional as F


class HopfieldLayer:
    def __init__(self, stored_patterns, beta=1.0, device="cpu"):
        """
        stored_patterns: [N, D] tensor or list of [D] tensors
        device: "cpu" or "cuda"
        """
        # normalize & move to device
        self.device = device

        # Accept lists/tuples of tensors, a single tensor, or array-like inputs
        if isinstance(stored_patterns, (list, tuple)):
            if len(stored_patterns) == 0:
                raise ValueError("stored_patterns must contain at least one pattern")
            # stack tensors or convert array-likes
            if isinstance(stored_patterns[0], torch.Tensor):
                stored = torch.stack(stored_patterns)
            else:
                stored = torch.tensor(stored_patterns)
        elif isinstance(stored_patterns, torch.Tensor):
            stored = stored_patterns
        else:
            stored = torch.tensor(stored_patterns)

        if stored.dim() == 1:
            stored = stored.unsqueeze(0)

        stored = stored.float().to(device)
        self.stored = F.normalize(stored, dim=1)
        self.beta = beta

    @torch.no_grad()
    def update(self, query):
        """
        query: tensor [D]
        returns: tensor [D]
        """
        q = F.normalize(query, dim=-1).to(self.device)

        # Correct shape: [1, D] @ [D, N] → [1, N]
        scores = self.beta * (q.unsqueeze(0) @ self.stored.T)[0]   # [N]

        weights = F.softmax(scores, dim=0)                         # [N]
        retrieved = weights.unsqueeze(0) @ self.stored             # [1, D]
        return retrieved.squeeze(0)

    @torch.no_grad()
    def energy(self, query):
        q = F.normalize(query, dim=-1).to(self.device)
        scores = self.beta * (q.unsqueeze(0) @ self.stored.T)[0]   # [N]
        return (-torch.logsumexp(scores, dim=0)).item()

    @torch.no_grad()
    def refine(self, query, max_steps=5, alpha=0.7, tol=1e-4):
        q = F.normalize(query, dim=-1).to(self.device)

        for _ in range(max_steps):
            retrieved = self.update(q)
            q_next = F.normalize(alpha*q + (1-alpha)*retrieved, dim=-1)

            delta = 1 - torch.dot(q, q_next).item()
            if delta < tol:
                break

            q = q_next
        return q
