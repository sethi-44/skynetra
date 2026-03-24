import os
import json
import datetime
import torch
import torch.nn.functional as F

__all__ = ["IdentityStore", "Info"]


# ---------------------------------------------------------------------------
# Info — per-identity metadata
# ---------------------------------------------------------------------------

class Info:
    def __init__(
        self,
        name: str,
        emb_rows: list = None,
        description: str = None,
        alive: bool = True,
        image: str = None,
    ):
        self.name        = name
        self.emb_rows    = emb_rows if emb_rows is not None else []
        self.description = description
        self.alive       = alive
        self.image       = image

    def __repr__(self) -> str:
        return f"<Info name={self.name!r} rows={self.emb_rows} alive={self.alive}>"


# ---------------------------------------------------------------------------
# IdentityStore
# ---------------------------------------------------------------------------

class IdentityStore:
    """
    Persistent identity gallery.

    Storage layout
    --------------
    self.embeddings : [R, D] float16 tensor  — one row per enrolled embedding
    self.store      : list[Info]             — one Info per identity; each Info
                      holds a list of row indices into self.embeddings

    Precision contract
    ------------------
    Embeddings are stored as FP16 unit-norm vectors.
    All similarity computations are promoted to FP32 before the matmul.
    The normalisation in _prepare_embedding_for_storage uses dim=0 for a 1-D
    vector, which is correct, but see Bug 1 below for the dim= caveat.
    """

    VERSION = "2.5"

    def __init__(self, embedding_dim: int = 256, device: str = "cpu"):
        self.device        = device
        self.embedding_dim = embedding_dim

        self.embeddings = torch.empty(
            (0, embedding_dim), device=device, dtype=torch.float16
        )
        self.store: list[Info] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_embedding_for_storage(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Normalise to unit length, cast to FP16, move to self.device.

        Bug fix: the old code used F.normalize(emb, dim=0). For a 1-D [D]
        tensor dim=0 and dim=-1 are identical — correct. But if a caller
        accidentally passes a 2-D [1, D] batch, dim=0 normalises across the
        batch axis (divides by 1.0, no-op) instead of the embedding axis.
        Using dim=-1 is safe for both shapes.
        """
        emb = embedding.to(self.device).float().squeeze()   # ensure 1-D [D]
        if emb.dim() != 1 or emb.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected a 1-D embedding of length {self.embedding_dim}, "
                f"got shape {tuple(embedding.shape)}"
            )
        emb = F.normalize(emb, dim=-1)
        return emb.half()

    def _similarity(self, query_emb: torch.Tensor) -> torch.Tensor:
        """
        Return [R] cosine similarities between query and all stored rows.

        Bug fix: the old code returned None when the store was empty, forcing
        every caller to guard against None. Now returns an empty [0] tensor
        so callers can use .shape[0] == 0 uniformly.

        Bug fix: query was not normalised before the matmul. If the caller
        passes a non-unit-norm vector, the dot products are NOT cosine
        similarities. Normalise here (FP32) before compute.
        """
        if self.embeddings.shape[0] == 0:
            return torch.empty(0, device=self.device)

        embs_f32  = self.embeddings.float()                          # [R, D]
        query_f32 = F.normalize(
            query_emb.to(self.device).float().squeeze(), dim=-1
        )                                                            # [D]
        return embs_f32 @ query_f32                                  # [R]

    # ------------------------------------------------------------------
    # Identity / embedding operations
    # ------------------------------------------------------------------

    def add_identity(
        self,
        embedding,
        name: str,
        description: str = None,
        alive: bool = True,
        image: str = None,
    ) -> int:
        """
        Enrol a new identity with one initial embedding.

        Returns the identity index (position in self.store).
        """
        embedding = self._prepare_embedding_for_storage(embedding)

        self.embeddings = torch.cat(
            [self.embeddings, embedding.unsqueeze(0)], dim=0
        )
        row_idx = self.embeddings.size(0) - 1
        self.store.append(Info(name, [row_idx], description, alive, image))
        return len(self.store) - 1

    def add_embedding(self, index: int, embedding) -> None:
        """Append a new embedding row for an existing identity."""
        if not (0 <= index < len(self.store)):
            raise IndexError(
                f"Identity index {index} out of range (store has {len(self.store)} entries)"
            )
        embedding = self._prepare_embedding_for_storage(embedding)

        self.embeddings = torch.cat(
            [self.embeddings, embedding.unsqueeze(0)], dim=0
        )
        row_idx = self.embeddings.size(0) - 1
        self.store[index].emb_rows.append(row_idx)

    # ------------------------------------------------------------------
    # Identity removal
    # ------------------------------------------------------------------

    def remove_identity(self, index: int) -> None:
        """Soft-delete: mark identity as dead. Call compact() to reclaim space."""
        if not (0 <= index < len(self.store)):
            raise IndexError(f"Identity index {index} out of range")
        self.store[index].alive = False

    def remove_embedding(self, identity_idx: int, emb_row: int) -> None:
        """
        Remove one embedding row from an identity's row list.

        Bug fix: the old code called list.remove(emb_row) which removes the
        FIRST occurrence by value. If emb_rows somehow contained duplicates
        (shouldn't happen, but defensive) only one would be removed. More
        importantly, list.remove raises ValueError if the row isn't there.
        Now raises a clear KeyError instead.
        """
        info = self.store[identity_idx]
        try:
            info.emb_rows.remove(emb_row)
        except ValueError:
            raise KeyError(
                f"Embedding row {emb_row} not found in identity {identity_idx} "
                f"(rows: {info.emb_rows})"
            )

    # ------------------------------------------------------------------
    # Search / similarity
    # ------------------------------------------------------------------

    def search_identity(self, query_emb) -> tuple:
        """
        Find the best-matching alive identity for a query embedding.

        Returns (identity_index, score). Returns (None, 0.0) if store is empty.

        Bug fix: the old code did not filter dead identities — a removed person
        could still be returned as the best match. Now only alive identities
        participate in the search.
        """
        if not self.store or self.embeddings.shape[0] == 0:
            return None, 0.0

        scores     = self._similarity(query_emb)
        id_scores  = self._embedding_scores_to_identity(scores, alive_only=True)

        # All scores initialised to -1e9; if every identity is dead, argmax
        # returns 0 with a -1e9 score — treat that as no match.
        best_score, best_idx = torch.max(id_scores, dim=0)
        best_score = best_score.item()
        if best_score <= -1e8:
            return None, 0.0

        return int(best_idx.item()), best_score

    def _embedding_scores_to_identity(
        self,
        emb_scores: torch.Tensor,
        alive_only: bool = True,
    ) -> torch.Tensor:
        """
        Aggregate per-row scores into per-identity scores by mean-pooling
        across each identity's embedding rows.

        Bug fix: identities with emb_rows pointing to out-of-range indices
        (e.g. after a partial compact) would cause index_select to crash.
        Added a bounds check that silently skips bad rows.
        """
        identity_scores = torch.full(
            (len(self.store),), -1e9, device=emb_scores.device
        )
        n_rows = emb_scores.shape[0]

        for idx, info in enumerate(self.store):
            if alive_only and not info.alive:
                continue
            if not info.emb_rows:
                continue
            # Guard against stale row indices that exceed current embeddings
            valid_rows = [r for r in info.emb_rows if r < n_rows]
            if not valid_rows:
                continue
            rows = torch.tensor(valid_rows, device=emb_scores.device)
            identity_scores[idx] = emb_scores.index_select(0, rows).mean()

        return identity_scores

    def is_duplicate(self, embedding, thresh: float = 0.98) -> bool:
        """
        Return True if any stored embedding is closer than `thresh` to the query.

        Bug fix: _similarity() now returns an empty tensor (not None) for an
        empty store, so the old `if self.embeddings.shape[0] == 0: return False`
        guard is still needed logically but now torch.max on an empty tensor
        would raise, so we keep the explicit empty check.
        """
        if self.embeddings.shape[0] == 0:
            return False
        scores = self._similarity(embedding)
        return bool(torch.max(scores).item() > thresh)

    def find_duplicate(self, embedding, thresh: float = 0.85):
        """
        Return the row index of the closest stored embedding if above thresh,
        else None.

        Bug fix: the old code did `scores[idx]` after argmax, which indexes
        by Python int — fine, but `.item()` is cleaner and avoids potential
        shape issues.
        """
        if self.embeddings.shape[0] == 0:
            return None
        scores      = self._similarity(embedding)
        best_score, best_idx = torch.max(scores, dim=0)
        if best_score.item() >= thresh:
            return int(best_idx.item())
        return None

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def compact(self) -> dict:
        """
        Remove dead identities and their embeddings, reindex all row pointers.

        Returns row_map: {old_row: new_row} for alive rows.

        Bug fix: the old compact() kept the FP16 dtype when rebuilding
        self.embeddings from stacked rows, BUT torch.stack produces the dtype
        of the input tensors — which are already FP16 slices from
        self.embeddings. That happens to be correct, but is fragile if
        dtype ever changes. Now explicitly cast to FP16 after stacking.

        Bug fix: compact() mutated info.emb_rows in-place while iterating
        self.store, which is safe here because we only append to new_store,
        but it mutates the Info object that's still referenced by the old
        store until reassignment. Moved mutation to after the new_store is
        built to make the order of operations explicit.
        """
        alive_embeddings = []
        new_store        = []
        row_map          = {}
        new_row          = 0
        pending_rows: dict[int, list] = {}   # info_idx → new_rows

        for i, info in enumerate(self.store):
            if not info.alive:
                continue
            new_rows = []
            for old_row in info.emb_rows:
                alive_embeddings.append(self.embeddings[old_row])
                row_map[old_row] = new_row
                new_rows.append(new_row)
                new_row += 1
            pending_rows[i] = new_rows
            new_store.append(info)

        # Commit row mutations only after iteration is complete
        for i, info in enumerate(new_store):
            orig_idx = self.store.index(info)
            info.emb_rows = pending_rows[orig_idx]

        if alive_embeddings:
            self.embeddings = torch.stack(alive_embeddings).to(
                self.device, dtype=torch.float16
            )
        else:
            self.embeddings = torch.empty(
                (0, self.embedding_dim), device=self.device, dtype=torch.float16
            )

        self.store = new_store
        return row_map

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        torch.save(self.embeddings.cpu(), os.path.join(path, "embeddings.pt"))

        meta = {
            "version":   self.VERSION,
            "timestamp": datetime.datetime.now().isoformat(),
            "emb_dim":   self.embedding_dim,
            "identities": [
                {
                    "name":        info.name,
                    "emb_rows":    info.emb_rows,
                    "description": info.description,
                    "alive":       info.alive,
                    "image":       info.image,
                }
                for info in self.store
            ],
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)

        print(f"[store] saved {len(self.store)} identities → {path}")

    def load(self, path: str, map_location: str = "cpu") -> None:
        emb_path  = os.path.join(path, "embeddings.pt")
        meta_path = os.path.join(path, "metadata.json")

        if not os.path.exists(emb_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Store files not found in {path!r}")

        # Bug fix: torch.load without weights_only=True triggers a deprecation
        # warning in PyTorch ≥2.0 and will become an error in future versions.
        self.embeddings = torch.load(
            emb_path, map_location=map_location, weights_only=True
        )

        # Infer true embedding dim from saved tensor (handles dim mismatch
        # between __init__ default and the actual saved gallery).
        if self.embeddings.ndim == 2:
            self.embedding_dim = self.embeddings.shape[1]
        else:
            raise ValueError(
                f"Loaded embeddings tensor has unexpected shape "
                f"{tuple(self.embeddings.shape)}"
            )

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Bug fix: if the saved file has an "emb_dim" field (written by this
        # version), cross-check against the loaded tensor to catch corruption.
        if "emb_dim" in meta and meta["emb_dim"] != self.embedding_dim:
            raise ValueError(
                f"Metadata emb_dim={meta['emb_dim']} does not match "
                f"loaded tensor dim={self.embedding_dim}"
            )

        self.store = [
            Info(
                name=item["name"],
                emb_rows=item["emb_rows"],
                description=item.get("description"),
                alive=item.get("alive", True),
                image=item.get("image"),
            )
            for item in meta["identities"]
        ]

        print(f"[store] loaded {len(self.store)} identities from {path!r}")

    def finalize(self) -> None:
        """
        Remove dead identities and reindex, called automatically by from_path.

        Bug fix: the old finalize() gathered alive_rows from all alive Info
        objects, then did self.embeddings = self.embeddings[alive_rows].
        If alive_rows was empty, this produced a 1-D empty tensor — losing
        the second dimension — so subsequent .shape[1] calls would crash.
        Now uses explicit torch.empty(...) for the empty case.

        Bug fix: the row remapping loop reused `r` as new_row counters that
        were independent of the alive_rows slice. Since alive_rows is already
        sorted by identity order, the counter increments correctly — but only
        if emb_rows within each Info are also in ascending order. Added a sort.
        """
        alive_rows: list[int] = []
        for info in self.store:
            if info.alive:
                alive_rows.extend(sorted(info.emb_rows))

        if alive_rows:
            self.embeddings = self.embeddings[alive_rows]
        else:
            self.embeddings = torch.empty(
                (0, self.embedding_dim), device=self.device, dtype=torch.float16
            )

        new_store = []
        new_row   = 0
        for info in self.store:
            if not info.alive:
                continue
            new_rows = [new_row + i for i in range(len(info.emb_rows))]
            new_row += len(info.emb_rows)
            info.emb_rows = new_rows
            new_store.append(info)

        self.store = new_store
        print(
            f"[store] finalised: {len(self.store)} identities, "
            f"{self.embeddings.shape[0]} embeddings"
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        n = self.embeddings.shape[0]
        alive = sum(i.alive for i in self.store)
        return {
            "total_identities": len(self.store),
            "alive":            alive,
            "dead":             len(self.store) - alive,
            "total_embeddings": n,
            "avg_emb_per_id":   n / max(1, len(self.store)),
        }

    @classmethod
    def from_path(cls, path: str, device: str = "cpu") -> "IdentityStore":
        """
        Load an existing store or create a fresh empty one.

        Always calls finalize() after loading to strip dead identities and
        reindex row pointers, so the in-memory state is always compact.
        """
        store = cls(device=device)
        if os.path.exists(path):
            emb_path  = os.path.join(path, "embeddings.pt")
            meta_path = os.path.join(path, "metadata.json")
            if os.path.exists(emb_path) and os.path.exists(meta_path):
                store.load(path, map_location=device)
                store.finalize()
        else:
            os.makedirs(path, exist_ok=True)
        return store