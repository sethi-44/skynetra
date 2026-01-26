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
        self.emb_rows = emb_rows or []
        self.description = description
        self.alive = alive
        self.image = image

    def __repr__(self):
        return f"<Info name={self.name} rows={self.emb_rows} alive={self.alive}>"


# ------------------------------
# Identity Store
# ------------------------------

class IdentityStore:
    VERSION = "2.4"

    def __init__(self, embedding_dim=256, device="cpu"):
        self.device = device
        self.embedding_dim = embedding_dim

        # FP16 STORAGE GUARANTEE
        self.embeddings = torch.empty(
            (0, embedding_dim),
            device=device,
            dtype=torch.float16
        )

        self.store = []


    # ---------------------------------------------------------
    # Internal helpers (precision contracts)
    # ---------------------------------------------------------

    def _prepare_embedding_for_storage(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        STORAGE BOUNDARY:
        - FP32 normalize
        - unit length
        - cast to FP16
        """
        emb = embedding.to(self.device).float()
        emb = F.normalize(emb, dim=0)
        return emb.half()

    def _similarity(self, query_emb: torch.Tensor) -> torch.Tensor:
        """
        COMPUTE BOUNDARY:
        - FP16 storage
        - FP32 compute
        """
        if self.embeddings.shape[0] == 0:
            return None

        embs_f32 = self.embeddings.float()
        query_f32 = query_emb.to(self.device).float()
        return torch.matmul(embs_f32, query_f32)


    # ---------------------------------------------------------
    # Identity / Embedding operations
    # ---------------------------------------------------------

    def add_identity(self, embedding, name, description=None, alive=True, image=None):
        embedding = self._prepare_embedding_for_storage(embedding)
        assert embedding.shape[-1] == self.embedding_dim, "wrong embedding size"

        self.embeddings = torch.cat(
            (self.embeddings, embedding.unsqueeze(0)), dim=0
        )

        row_idx = self.embeddings.size(0) - 1
        info = Info(name, [row_idx], description, alive, image)
        self.store.append(info)

        return len(self.store) - 1

    def add_embedding(self, index, embedding):
        if not (0 <= index < len(self.store)):
            raise IndexError("identity index out of range")

        embedding = self._prepare_embedding_for_storage(embedding)
        assert embedding.shape[-1] == self.embedding_dim, "wrong embedding size"

        self.embeddings = torch.cat(
            (self.embeddings, embedding.unsqueeze(0)), dim=0
        )

        row_idx = self.embeddings.size(0) - 1
        self.store[index].emb_rows.append(row_idx)


    # ---------------------------------------------------------
    # Identity removal
    # ---------------------------------------------------------

    def remove_identity(self, index):
        self.store[index].alive = False

    def remove_embedding(self, identity_idx, emb_row):
        info = self.store[identity_idx]
        info.emb_rows.remove(emb_row)


    # ---------------------------------------------------------
    # Search / similarity
    # ---------------------------------------------------------

    def search_identity(self, query_emb):
        if self.embeddings.shape[0] == 0:
            return None, 0.0

        scores = self._similarity(query_emb)
        id_scores = self._embedding_scores_to_identity(scores)

        best_idx = torch.argmax(id_scores).item()
        return best_idx, id_scores[best_idx].item()

    def _embedding_scores_to_identity(self, emb_scores: torch.Tensor):
        identity_scores = torch.full(
            (len(self.store),),
            -1e9,
            device=emb_scores.device
        )

        for idx, info in enumerate(self.store):
            if not info.emb_rows:
                continue
            rows = torch.tensor(info.emb_rows, device=emb_scores.device)
            identity_scores[idx] = emb_scores.index_select(0, rows).mean()

        return identity_scores

    def is_duplicate(self, embedding, thresh=0.98):
        if self.embeddings.shape[0] == 0:
            return False

        scores = self._similarity(embedding)
        return torch.max(scores).item() > thresh

    def find_duplicate(self, embedding, thresh=0.85):
        if self.embeddings.shape[0] == 0:
            return None

        scores = self._similarity(embedding)
        idx = torch.argmax(scores).item()
        return idx if scores[idx] >= thresh else None


    # ---------------------------------------------------------
    # Compaction
    # ---------------------------------------------------------

    def compact(self):
        alive_embeddings = []
        new_store = []
        row_map = {}
        new_row = 0

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

        if alive_embeddings:
            self.embeddings = torch.stack(alive_embeddings).to(self.device)
        else:
            self.embeddings = torch.empty(
                (0, self.embedding_dim),
                device=self.device,
                dtype=torch.float16
            )

        self.store = new_store
        return row_map


    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        torch.save(
            self.embeddings.cpu(),
            os.path.join(path, "embeddings.pt")
        )

        meta = {
            "version": self.VERSION,
            "timestamp": datetime.datetime.now().isoformat(),
            "identities": [
                {
                    "name": info.name,
                    "emb_rows": info.emb_rows,
                    "description": info.description,
                    "alive": info.alive,
                    "image": info.image,
                }
                for info in self.store
            ]
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)

        print(f"💾 saved {len(self.store)} identities → {path}")

    def load(self, path, map_location="cpu"):
        self.embeddings = torch.load(
            os.path.join(path, "embeddings.pt"),
            map_location=map_location
        )

        # 🔒 INFER TRUE EMBEDDING DIM
        self.embedding_dim = self.embeddings.shape[1]

        with open(os.path.join(path, "metadata.json"), "r") as f:
            meta = json.load(f)

        self.store = []
        for item in meta["identities"]:
            self.store.append(
                Info(
                    name=item["name"],
                    emb_rows=item["emb_rows"],
                    description=item.get("description"),
                    alive=item.get("alive", True),
                    image=item.get("image"),
                )
            )

        print(f"📂 loaded {len(self.store)} identities from {path}")

    def finalize(self):
        alive_rows = []
        for info in self.store:
            if info.alive:
                alive_rows.extend(info.emb_rows)

        self.embeddings = self.embeddings[alive_rows]

        new_store = []
        row_map = {}
        new_row = 0

        for info in self.store:
            if info.alive:
                new_rows = []
                for r in info.emb_rows:
                    new_rows.append(new_row)
                    row_map[r] = new_row
                    new_row += 1
                info.emb_rows = new_rows
                new_store.append(info)

        self.store = new_store
        print(
            f"🧹 Compacted: {len(self.store)} identities, "
            f"{self.embeddings.shape[0]} embeddings"
        )


    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def stats(self):
        return {
            "total_identities": len(self.store),
            "alive": sum(i.alive for i in self.store),
            "dead": sum(not i.alive for i in self.store),
            "total_embeddings": self.embeddings.shape[0],
            "avg_emb_per_id": self.embeddings.shape[0] / max(1, len(self.store)),
        }

    @classmethod
    def from_path(cls, path, device="cpu"):
        store = cls(device=device)
        if os.path.exists(path):
            if (
                os.path.exists(os.path.join(path, "embeddings.pt"))
                and os.path.exists(os.path.join(path, "metadata.json"))
            ):
                store.load(path, map_location=device)
                store.finalize()
        else:
            os.makedirs(path, exist_ok=True)
        return store
