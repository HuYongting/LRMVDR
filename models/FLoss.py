import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def focal_loss(self, logits, labels, gamma=2, reduction="mean"):

        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        log_pt = -ce_loss
        pt = torch.exp(log_pt)
        weights = (1 - pt) ** gamma
        fl = weights * ce_loss

        if reduction == "sum":
            fl = fl.sum()
        elif reduction == "mean":
            fl = fl.mean()
        else:
            raise ValueError(f"reduction '{reduction}' is not valid")
        return fl

    def forward(self, logits, labels):
        return self.focal_loss(logits, labels, gamma=self.gamma)



