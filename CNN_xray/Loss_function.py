import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        # logits: tensor of shape (N, 118, H, W), raw model outputs
        # targets: tensor of shape (N, 118, H, W), binary ground truth masks (0 or 1)
        prob = torch.sigmoid(logits)  # convert logits to probabilities in [0,1]
        # Compute intersection and union (sums) per class for each sample
        dims = (2, 3)  # dimensions to sum over (H, W)
        intersection = (prob * targets).sum(dim=dims)  # (N, 118)
        union = prob.sum(dim=dims) + targets.sum(dim=dims)  # (N, 118)
        # Compute dice score per class per sample
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (N, 118)
        # Average dice score across classes and batch
        dice_score = dice_score.mean()  
        return 1.0 - dice_score  # (1 - mean dice score)

class GeneralisedDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)

        preds = preds.flatten(2)
        targets = targets.flatten(2)


        gsum = targets.sum(-1)                    
        w = 1.0 / (gsum**2 + self.eps)           
        inter = (preds * targets).sum(-1)

        num = (w * inter).sum(1)
        den = (w * (preds.sum(-1) + gsum)).sum(1)
        gdl = 1 - 2 * num / (den + self.eps)
        return gdl.mean()


def focal_tversky_loss(logits, targets, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6, ignore_empty=True):
    """
    logits: Tensor of shape (B, C, H, W) - raw model outputs for C classes.
    targets: Tensor of shape (B, C, H, W) - ground truth masks (0 or 1).
    """
    probs = torch.sigmoid(logits)                     # convert to [0,1] probability
    dims = (2, 3)                                     # dimensions to sum over (H, W)
    # calculate true positives, false positives, false negatives per class
    TP = (probs * targets).sum(dim=dims)
    FP = (probs * (1 - targets)).sum(dim=dims)
    FN = ((1 - probs) * targets).sum(dim=dims)
    # Tversky index per class
    TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    if ignore_empty:
        # Mask out classes that have no foreground (targets sum to 0) or all foreground (targets sum = H*W)
        empty_mask = (targets.sum(dim=dims) == 0) | (targets.sum(dim=dims) == targets.shape[2]*targets.shape[3])
        TI[empty_mask] = 1.0  # set TI=1 so that (1-TI)=0, no loss for empty/full classes
    # Focal Tversky Loss: focus on classes with low TI
    FTL = (1 - TI) ** gamma
    return FTL.mean()  # average over classes and batch
