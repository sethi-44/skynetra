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



def identify_person(refined, gallery, id_names,delta,threshold=0.8,delta_threshold=0.2):
    """
    cosine / dot similarity over gallery identities
    returns best match + similarity score
    """
    print("Delta:", delta)
    if delta < delta_threshold:
        return "Unknown", 0.0
    # Ensure both tensors are on the same device as `refined` to avoid CPU/CUDA mismatch
    if hasattr(refined, 'device'):
        gallery = gallery.to(refined.device)
        refined = refined.to(refined.device)
    best_name="Unknown"
    best_score=0.0
    scores = torch.matmul(gallery, refined)
    best_idx = scores.argmax().item()
    best_name = id_names[best_idx]
    best_score = scores[best_idx].item()
    if best_score<threshold:
        best_name="Unknown"
    return best_name, best_score