import torch
import torch.nn.functional as F


class HopfieldLayer:
    def __init__(self, stored_patterns, beta=1.0, device="cpu"):
        """
        stored_patterns: [N, D] tensor
        device: "cpu" or "cuda"
        """
        # normalize & move to device
        self.device = device
        self.stored = F.normalize(stored_patterns, dim=1).to(device)
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
    def refine(self, query, max_steps=5, alpha=0.5, tol=1e-4):
        q = F.normalize(query, dim=-1).to(self.device)

        for _ in range(max_steps):
            retrieved = self.update(q)
            q_next = F.normalize(alpha*q + (1-alpha)*retrieved, dim=-1)

            delta = 1 - torch.dot(q, q_next).item()
            if delta < tol:
                break

            q = q_next
        return q
