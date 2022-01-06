import torch.nn as nn
import torch.nn.functional as F
import torch
def kl_div(ps, qt):
    eps = 1e-10
    ps = ps + eps
    qt = qt + eps
    loss = qt * torch.log(qt) - qt * torch.log(ps)
    return loss.sum(1)


def kl_bce(ps, qt, label):
    # ps, qt: N*80 ==> N*80*2 for [0, 1]
    eps = 1e-10

    ps_1 = 1.0 - ps #N*80
    ps = torch.stack((ps, ps_1), dim=2)

    qt_1 = 1.0 - qt
    qt = torch.stack((qt, qt_1), dim=2)

    n_s = list(range(0, label.size(0), 1))
    ps[n_s,label] = 1.0 - ps[n_s,label] # N*80*2
    qt[n_s,label] = 1.0 - qt[n_s,label] # N*80*2

    ps = torch.clamp(ps, min=eps)
    qt = torch.clamp(qt, min=eps)
    # compute kl alond dim=2
    loss = qt * torch.log(qt) - qt * torch.log(ps)
    loss = loss.sum(2).sum(1) / ps.size(1)

    return loss

def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.
    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
            T * T)

    return kd_loss

def kl_ce(pred,target, T,detach_target=True):
    assert pred.size() == target.size()
    pred = pred.simoid()
    target = target.sigmoid()
    pred = F.normalize(pred, dim=1, p=1)
    target = F.normalize(target, dim=1, p=1)
    loss = kl_div(pred, target)
    return None
