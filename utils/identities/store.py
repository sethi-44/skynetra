import os
import json
import datetime
import torch
import torch.nn.functional as F

__all__ = ["IdentityStore", "Info"]

# ------------------------------
# Metadata for identities
# ------------------------------

class Info:
    def __init__(self, name, emb_rows=None, description=None, alive=True, image=None):
        self.name = name
        self.emb_rows = emb_rows or []     # list of embedding row indices
        self.description = description
        self.alive = alive
        self.image = image                 # path or None

    def __repr__(self):
        return f"<Info name={self.name} rows={self.emb_rows} alive={self.alive}>"


# ------------------------------
# Identity Store
# ------------------------------

class IdentityStore:
    VERSION = "2.3"

    def __init__(self, embedding_dim=128, device="cpu"):
        self.device = device
        self.embedding_dim = embedding_dim

        self.store = []                                 # list[Info]
        self.embeddings = torch.empty((0, embedding_dim), device=device)

    # ---------------------------------------------------------
    # Identity / Embedding operations
    # ---------------------------------------------------------

    def add_identity(self, embedding, name, description=None, alive=True, image=None):
        """
        Create a new identity with its FIRST embedding.
        Returns the identity index.
        """
        embedding = embedding.to(self.device)
        assert embedding.shape[-1] == self.embedding_dim, "wrong embedding size"

        self.embeddings = torch.cat((self.embeddings, embedding.unsqueeze(0)), dim=0)
        row_idx = self.embeddings.size(0) - 1

        info = Info(name, [row_idx], description, alive, image)
        self.store.append(info)
        return len(self.store) - 1

    def add_embedding(self, index, embedding):
        """Add additional embedding to an existing identity."""
        if not (0 <= index < len(self.store)):
            raise IndexError("identity index out of range")

        embedding = embedding.to(self.device)
        assert embedding.shape[-1] == self.embedding_dim, "wrong embedding size"

        self.embeddings = torch.cat((self.embeddings, embedding.unsqueeze(0)), dim=0)
        row_idx = self.embeddings.size(0) - 1

        self.store[index].emb_rows.append(row_idx)

    def remove_identity(self, index):
        """Soft-delete identity (keeps data until compact)."""
        if not (0 <= index < len(self.store)):
            raise IndexError("identity index out of range")
        self.store[index].alive = False

    def remove_embedding(self, identity_idx, emb_row):
        """Remove a single embedding index from an identity."""
        info = self.get_identity(identity_idx)
        if emb_row in info.emb_rows:
            info.emb_rows.remove(emb_row)
        else:
            raise ValueError("embedding does not belong to identity")

    def get_identity(self, index):
        if not (0 <= index < len(self.store)):
            raise IndexError("identity index out of range")
        return self.store[index]

    # ---------------------------------------------------------
    # Compaction / clean-up
    # ---------------------------------------------------------

    def compact(self):
        """
        Hard cleanup:
        - removes dead identities
        - rebuilds embeddings
        - updates indexes
        """
        alive_embeddings = []
        new_store = []
        row_map = {}
        new_row = 0

        # rebuild embeddings & remap indices
        for info in self.store:
            if not info.alive:
                continue

            new_rows = []
            for old_row in info.emb_rows:
                alive_embeddings.append(self.embeddings[old_row])
                new_rows.append(new_row)
                row_map[old_row] = new_row
                new_row += 1

            info.emb_rows = new_rows
            new_store.append(info)

        # rebuild tensors
        if len(alive_embeddings) > 0:
            self.embeddings = torch.stack(alive_embeddings, dim=0).to(self.device)
        else:
            self.embeddings = torch.empty((0, self.embedding_dim), device=self.device)

        self.store = new_store
        return row_map

    # ---------------------------------------------------------
    # Device transfer
    # ---------------------------------------------------------

    def to(self, device):
        self.device = device
        self.embeddings = self.embeddings.to(device)
        return self

    # ---------------------------------------------------------
    # Identity search / scoring
    # ---------------------------------------------------------

    def search_identity(self, query_emb):
        """
        Returns: (identity_idx, score)
        """
        if len(self.embeddings) == 0:
            return None, 0.0

        query_emb = query_emb.to(self.device)
        scores = torch.matmul(self.embeddings, query_emb)

        id_scores = self.conv_emb_score_to_identity_score(scores)
        best_idx = torch.argmax(id_scores).item()
        return best_idx, id_scores[best_idx].item()

    def conv_emb_score_to_identity_score(self, emb_scores: torch.Tensor):
        """
        Convert embedding scores → identity scores by averaging per identity.
        """
        identity_scores = torch.zeros(len(self.store), device=emb_scores.device)

        for identity_idx, info in enumerate(self.store):
            if len(info.emb_rows) == 0:
                identity_scores[identity_idx] = -1e9
                continue

            rows = torch.tensor(info.emb_rows, device=emb_scores.device)
            identity_scores[identity_idx] = emb_scores.index_select(0, rows).mean()

        return identity_scores

    def find_duplicate(self, embedding, thresh=0.85):
        """Check if embedding already exists in store."""
        if len(self.embeddings) == 0:
            return None

        embedding = embedding.to(self.device)
        scores = torch.matmul(self.embeddings, embedding)
        idx = torch.argmax(scores).item()

        if scores[idx] >= thresh:
            return idx
        return None

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        torch.save(self.embeddings.cpu(), os.path.join(path, "embeddings.pt"))

        meta = {
            "version": self.VERSION,
            "timestamp": datetime.datetime.now().isoformat(),
            "identities": []
        }

        for info in self.store:
            meta["identities"].append({
                "name": info.name,
                "emb_rows": info.emb_rows,
                "description": info.description,
                "alive": info.alive,
                "image": info.image
            })

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)

        print(f"💾 saved {len(self.store)} identities → {path}")

    def load(self, path, map_location="cpu"):
        emb_path = os.path.join(path, "embeddings.pt")
        meta_path = os.path.join(path, "metadata.json")

        if not os.path.exists(emb_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("missing embeddings.pt or metadata.json")

        self.embeddings = torch.load(emb_path, map_location=map_location)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if meta.get("version") != self.VERSION:
            print(f"⚠ version mismatch stored={meta.get('version')} code={self.VERSION}")

        self.store = []
        for item in meta["identities"]:
            info = Info(
                name=item["name"],
                emb_rows=item["emb_rows"],
                description=item.get("description"),
                alive=item.get("alive", True),
                image=item.get("image")
            )
            self.store.append(info)

        print(f"📂 loaded {len(self.store)} identities from {path}")

    def finalize(self):
        """
        Rebuild embedding tensor to keep only alive identities.
        Must call after load() if you want a clean state.
        """
        alive_rows = []
        for info in self.store:
            if info.alive:
                alive_rows.extend(info.emb_rows)

        # rebuild embeddings
        new_embeddings = self.embeddings[alive_rows]

        # rebuild new row indices
        new_store = []
        row_map = {}
        new_row_idx = 0
        for info in self.store:
            if info.alive:
                new_rows = []
                for r in info.emb_rows:
                    if r in alive_rows:
                        new_rows.append(new_row_idx)
                        row_map[r] = new_row_idx
                        new_row_idx += 1
                info.emb_rows = new_rows
                new_store.append(info)

        self.embeddings = new_embeddings
        self.store = new_store
        print(f"🧹 Compacted: now {len(self.store)} identities, {self.embeddings.shape[0]} embeddings")
    def is_duplicate(self, embedding, thresh=0.98):
        if self.embeddings.shape[0] == 0:
            return False
        sims = torch.matmul(self.embeddings, embedding)
        return torch.max(sims).item() > thresh
    def merge_identities(self, idx_a, idx_b, new_name=None):
        # append embeddings
        self.store[idx_a].emb_rows.extend(self.store[idx_b].emb_rows)
        # kill second
        self.store[idx_b].alive = False
        # rename
        if new_name: self.store[idx_a].name = new_name
    def stats(self):
        return {
            "total_identities": len(self.store),
            "alive": sum(info.alive for info in self.store),
            "dead": sum(not info.alive for info in self.store),
            "total_embeddings": self.embeddings.shape[0],
            "avg_emb_per_id": self.embeddings.shape[0] / max(1,len(self.store)),
        }
    @classmethod
    def from_path(cls, path, device="cpu"):
        store = cls(device=device)
        if os.path.exists(path):
            emb_path = os.path.join(path, "embeddings.pt")
            meta_path = os.path.join(path, "metadata.json")
            # Only load if both files exist
            if os.path.exists(emb_path) and os.path.exists(meta_path):
                store.load(path, map_location=device)
                store.finalize()
        else:
            os.makedirs(path, exist_ok=True)
        return store

    

    
    
    


