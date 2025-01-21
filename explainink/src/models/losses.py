import torch
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F


def attention_kl_loss(scores: T, nll_loss: T, eps: float = 1e-8) -> T:
    """
    Computes KL divergence between attention scores and
    the uniform distribution. The purpose is to maximize
    the KL divergence between them, to avoid uniform attention scores.

    Args:
        scores (Tensor): attention scores wit shape (batch, length)
        nll_loss (Tensor): NLL loss to clamp de KL divergence (scalar)

    Returns:
        Tensor: clamped KL divergence (scalar)
    """
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    uniform = torch.ones_like(scores) / scores.shape[-1]
    clamped_target = torch.clamp(scores, min=eps)
    clamped_uniform = torch.clamp(uniform, min=eps)
    attention_loss = -kl_loss(clamped_uniform, clamped_target)
    # Clamp the KL to the NLL.
    # Tries to avoid the models to focus just on reducing the KL
    # when its magnitude is much larger than the NLL loss.
    clamped_attention_loss = torch.clamp(attention_loss, max=nll_loss.item())
    return clamped_attention_loss


def conicity_loss(last_hidden_states: T, alpha: float = 0.3) -> T:
    """
    Conicity loss for diversity driven training from:
    https://aclanthology.org/2020.acl-main.387.pdf

    Args:
        last_hidden_states (Tensor): hidden states from the last encoder layer.
        alpha (float): value to scale the conicity loss.

    Returns:
        Tensor: scaled conicity loss (scalar)
    """
    mean_rows = last_hidden_states.mean(dim=1)
    conicity = F.cosine_similarity(
        last_hidden_states, mean_rows.unsqueeze(1), dim=-1
    )
    return alpha * conicity.mean()
