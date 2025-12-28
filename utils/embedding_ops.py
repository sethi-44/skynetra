import torch
import torch.nn.functional as F
from utils.hopfield_layer import HopfieldLayer

def normalize_embeddings(embs):
    return F.normalize(embs, dim=1)

def pool_embeddings(buf,device):
    """
    Hopfield-pooling over buffer embeddings
    turns raw temporal embeddings into a stable identity representation
    """
    buf_tensor = torch.stack(buf).to(device)
    mean_init = buf_tensor.mean(dim=0)
    hop_buf = HopfieldLayer(buf_tensor, device=device)
    pooled = hop_buf.refine(mean_init)
    return pooled
    
def refine_identity(pooled, hop):
    """
    Refinement using global identity memory Hopfield
    returns refined embedding + energy for confidence
    """
    refined = hop.refine(pooled)
    energy = hop.energy(refined)
    return refined, energy


def identify_person(refined, gallery, id_names):
    """
    cosine / dot similarity over gallery identities
    returns best match + similarity score
    """
    # Ensure both tensors are on the same device as `refined` to avoid CPU/CUDA mismatch
    if hasattr(refined, 'device'):
        gallery = gallery.to(refined.device)
        refined = refined.to(refined.device)

    scores = torch.matmul(gallery, refined)
    best_idx = scores.argmax().item()
    best_name = id_names[best_idx]
    best_score = scores[best_idx].item()
    return best_name, best_score