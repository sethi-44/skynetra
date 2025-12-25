import torch
import torch.nn.functional as F

def normalize_embeddings(embs):
    return F.normalize(embs, dim=1)

def mean_pool_embeddings(emb_list):
    if len(emb_list) == 0:
        return None
    
    E = torch.stack(emb_list)
    pooled = F.normalize(E.mean(dim=0), dim=0)
    return pooled

def identify(pooled_emb, gallery, threshold=0):
    best_name, best_score = "Unknown", -1.0

    for name, ref_emb in gallery.items():
        score = torch.dot(pooled_emb, ref_emb).item()
        if score > best_score:
            best_name, best_score = name, score

    if best_score < threshold:
        return "Unknown", best_score

    return best_name, best_score
