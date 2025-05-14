import torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from data import XRaySegmentationDataset
from model import MultiLabelSegModel
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast
# ---------------- CONFIG -----------------
DATA_ROOT   = Path("/home/user/yibo_workspace")          
TRAIN_IMG   = DATA_ROOT/"images/train"
TRAIN_MSK   = DATA_ROOT/"masks/train"
VAL_IMG     = DATA_ROOT/"images/eval"
VAL_MSK     = DATA_ROOT/"masks/eval"

BATCH_VAL   = 4
BATCH_TRAIN = 4

# val: 49 lung_upper_lobe_right 48 lung_upper_lobe_left 47 lung_middle_lobe_right 46 lung_lower_lobe_right 45 lung_lower_lobe_left 
# -----------------------------------------
@torch.no_grad()
def per_class_dice(prob, gt, skip_bg=True):
    """
    prob, gt : tensors (B, C, H, W)  with prob in [0,1], gt 0/1
    Returns   : list of (class_idx, dice_value)  length ≤ C-1
    Soft Dice, no threshold, ignores classes whose union == 0.
    """
    dices = []
    C = prob.shape[1]
    for c in range(C):
        if skip_bg and c == 0:
            continue
        p = prob[:, c]
        g = gt[:, c]
        union = p.sum() + g.sum()
        if union == 0:
            continue              # absent everywhere
        inter = (p * g).sum()
        dice_c = (2 * inter / (union + 1e-6)).item()
        dices.append((c, dice_c))
    return dices

def visualize_class(img, prob, gt, cls, *, cmap="viridis",
                    thr=0.5, save_path=None):
    """
    img  : tensor (1,H,W)  – pre‑processed X‑ray
    prob : tensor (C,H,W)  – model probabilities in [0,1]
    gt   : tensor (C,H,W)  – ground‑truth 0/1
    cls  : int             – class index to show
    thr  : float           – threshold for binary contour
    """

    xray = TF.to_pil_image(img.cpu().squeeze(0))        # grayscale
    pred = prob[cls].detach().cpu()
    mask = gt[cls].cpu()

    plt.figure(figsize=(9,3))

    # 1. X‑ray only
    plt.subplot(1,3,1)
    plt.imshow(xray, cmap="gray"); plt.axis("off"); plt.title("X‑ray")

    # 2. Prediction heat‑map overlay
    plt.subplot(1,3,2)
    plt.imshow(xray, cmap="gray")
    plt.imshow(pred, vmin=0, vmax=1, cmap=cmap, alpha=.5)
    plt.axis("off"); plt.title(f"Prediction  (class {cls})")

    # 3. GT contour
    plt.subplot(1,3,3)
    plt.imshow(xray, cmap="gray")
    plt.contour(mask.numpy() > thr, levels=[0.5], colors="red", linewidths=1)
    plt.axis("off"); plt.title("GT mask")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[✓] saved → {save_path}")
        plt.close()
    else:
        plt.show()

def show_binary_masks(img, prob, gt, gt_cls, pred_cls, *, thr=0.5, save_path=None):
    """
    prob : (C,H,W)   – network probabilities in [0,1]
    gt   : (C,H,W)   – ground‑truth 0/1
    cls  : int       – class index to visualise
    thr  : float     – threshold for prediction → binary
    """

    # combine all the classes in the pred_cls into one binary mask
    pred_bin = torch.zeros_like(prob[0], dtype=torch.bool)
    for cls in pred_cls:
        pred_bin = pred_bin | (prob[cls] > thr)
    pred_bin = pred_bin.cpu()
    gt_bin   = (gt[gt_cls] > 0.5).cpu()          # gt already 0/1, keep for safety
    xray = TF.to_pil_image((img.cpu().squeeze(0) * 0.229 + 0.485).clamp(0,1))

    plt.figure(figsize=(24,12))
    plt.subplot(1,3,1); plt.imshow(gt_bin, cmap="gray");  plt.axis("off"); plt.title(f"GT class {gt_cls}")
    plt.subplot(1,3,2); plt.imshow(pred_bin, cmap="gray");plt.axis("off"); plt.title(f"Pred ≥{thr}")
    plt.subplot(1,3,3); plt.imshow(xray, cmap="gray"); plt.axis("off"); plt.title(f"X-ray")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[✓] saved → {save_path}")
        plt.close()
    else:
        plt.show()

def merge_prob_max(prob, classes):
    # pick the class subset → (B,len,K,H,W) then max over that dim
    merged = prob[:, classes, :, :].max(dim=1, keepdim=True).values
    return merged

def macro_soft_dice(pred, gt, skip_bg=True):
    dices = []
    for c in range(pred.shape[1]):
        if skip_bg and c == 0:
            continue
        g = gt[:, c]
        if g.sum() == 0:
            continue            # absent class
        p = pred[:, c]
        inter = (p * g).sum()
        union = p.sum() + g.sum()
        dices.append((2*inter / (union + 1e-6)).item())
    return np.mean(dices) if dices else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # datasets & loaders
    train_ds = XRaySegmentationDataset(TRAIN_IMG, TRAIN_MSK, transform=False)
    val_ds   = XRaySegmentationDataset(VAL_IMG,   VAL_MSK,   transform=False, eval=True)
    train_dl = DataLoader(train_ds, BATCH_TRAIN, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   BATCH_VAL,   shuffle=False,
                          num_workers=4, pin_memory=True)

    ckpt = torch.load("checkpoints/best_v2.pt", map_location=device, weights_only=False)
    model = MultiLabelSegModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # -------------------------- Output the Dice score for each class --------------------------
    # class_dice = {}

    # for img, msk in val_dl:
    #     img, msk = img.to(device), msk.to(device)
    #     logits = model(img)
    #     prob = torch.sigmoid(logits)
    #     dices = per_class_dice(prob, msk)
    #     for c, dice in dices:
    #         class_dice.setdefault(c, []).append(dice)
        
    # table = [(c, np.mean(vals)) for c, vals in class_dice.items()]
    # table.sort(key=lambda x: x[1])      # ascending
    # print("Lowest‑Dice classes:")
    # for c,d in table[:10]:
    #     print(f" class {c:<3d}  Dice {d:.3f}")
    # print("Highest‑Dice classes:")
    # for c,d in table[-10:]:
    #     print(f" class {c:<3d}  Dice {d:.3f}")
    # print(f"Total classes: {len(table)}")

    # -------------------------- Visualise the predictions --------------------------
    for batch_idx, (img, msk) in enumerate(val_dl):
        img, msk = img.to(device), msk.to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(img))

        example_idx = 0
        if batch_idx == 0:
            # print("Shape: ", img[1].shape, prob[1].shape, msk[1].shape)
            show_binary_masks(img[example_idx], prob[example_idx], msk[example_idx], gt_cls=1, pred_cls=[45, 48], thr=0.5,
                  save_path="runs/vis/eval_left_lung.png")
            show_binary_masks(img[example_idx], prob[example_idx], msk[example_idx], gt_cls=0, pred_cls=[46, 47, 49], thr=0.5,
                  save_path="runs/vis/eval_right_lung.png")
            show_binary_masks(img[example_idx], prob[example_idx], msk[example_idx], gt_cls=2, pred_cls=[28], thr=0.5,
                  save_path="runs/vis/eval_heart.png")
            break          # only one demo

    # -------------------------- Calculate the metrics --------------------------
    sq_err_sum  = 0.0          # ∑ e²           ┐  for MSE / σ
    sq_err4_sum = 0.0          # ∑ e⁴           ┘
    pix         = 0            # # pixels used

    dice_sum    = 0.0          # ∑ D            ┐  for Dice / σ
    dice_sq_sum = 0.0          # ∑ D²           ┘
    dice_n      = 0            # # Dice values (images × organs)

    eps = 1e-7                 # numerical stabiliser for Dice

    model.eval()
    with torch.no_grad(), autocast():
        for img, msk in val_dl:
            img, msk = img.to(device), msk.to(device)
            prob = torch.sigmoid(model(img))

            # ---------------- organ‑wise predictions ----------------------------
            left_prob   = merge_prob_max(prob, [45, 48])            # (B,1,H,W)
            right_prob  = merge_prob_max(prob, [46, 47, 49])
            heart_prob  = merge_prob_max(prob, [28])

            # ground‑truth masks – add channel dim so shapes match (B,1,H,W)
            left_gt   = msk[:, 1, :, :].unsqueeze(1)
            right_gt  = msk[:, 0, :, :].unsqueeze(1)
            heart_gt  = msk[:, 2, :, :].unsqueeze(1)

            # --------------- helpers to update running statistics ---------------
            def _acc_mse(pred, target):
                nonlocal sq_err_sum, sq_err4_sum, pix
                e2 = (pred - target).pow(2)
                sq_err_sum  += e2.sum().item()
                sq_err4_sum += e2.pow(2).sum().item()
                pix         += e2.numel()

            def _acc_dice(pred, target):
                """
                Soft Dice per image (batch) then accumulate mean & mean‑square.
                """
                nonlocal dice_sum, dice_sq_sum, dice_n
                # flatten each image into a vector
                p = pred.view(pred.size(0), -1)
                t = target.view(target.size(0), -1)

                inter = (p * t).sum(1) * 2          # 2·|P∩T|
                denom = p.sum(1) + t.sum(1) + eps   # |P|+|T|
                dice  = (inter + eps) / denom       # (2|∩| + eps)/( |P|+|T| + eps )

                dice_sum    += dice.sum().item()
                dice_sq_sum += (dice.pow(2)).sum().item()
                dice_n      += dice.numel()         # add B

            # update for all three organs
            for pred, gt in [(left_prob, left_gt),
                            (right_prob, right_gt),
                            (heart_prob, heart_gt)]:
                _acc_mse(pred, gt)
                _acc_dice(pred, gt)

    # ------------------ final statistics ----------------------------------------
    mse = sq_err_sum / pix
    mse_var = max(sq_err4_sum / pix - mse**2, 0.0)
    mse_std = mse_var ** 0.5

    dice_mean = dice_sum / dice_n
    dice_var  = max(dice_sq_sum / dice_n - dice_mean**2, 0.0)
    dice_std  = dice_var ** 0.5

    print(f"MSE  = {mse:.6f}  |  σ(MSE)  = {mse_std:.6f}")
    print(f"Dice = {dice_mean:.6f} |  σ(Dice) = {dice_std:.6f}")





if __name__ == "__main__":
    main()