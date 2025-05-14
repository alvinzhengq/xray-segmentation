#!/usr/bin/env python3
# train_full.py  –  scalable GPU version
# -----------------------------------------------------------
import random, shutil
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random

# ---------- local modules ----------
from data        import XRaySegmentationDataset   # new Dataset class
from model          import MultiLabelSegModel
from Loss_function  import focal_tversky_loss
# ------------------------------------

# ---------------- CONFIG -----------------
DATA_ROOT   = Path("/home/user/yibo_workspace")         
TRAIN_IMG   = DATA_ROOT/"images/all_train_data"
TRAIN_MSK   = DATA_ROOT/"masks/all_train_data"
VAL_IMG     = DATA_ROOT/"images/eval"
VAL_MSK     = DATA_ROOT/"masks/eval"

BATCH_TRAIN = 4          # fit GPU mem
BATCH_VAL   = 4
EPOCHS      = 100
LR          = 3e-4 #1e-3
ENC_LR      = 3e-5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIR = Path("checkpoints"); CKPT_DIR.mkdir(exist_ok=True)
VIS_DIR  = Path("runs/vis");   VIS_DIR.mkdir(exist_ok=True, parents=True)
PLOT_CH  = [0, 1, 2]         # channels to visualise on val
# -----------------------------------------

# ---------- helper: macro soft Dice ----------
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
# --------------------------------------------

def save_best(model, optim, epoch, dice, train_loss, val_loss):
    path = CKPT_DIR / "best.pt"
    torch.save({"epoch": epoch,
                "dice": dice,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss}, path)
    print(f"[✓] best checkpoint updated  (Dice={dice:.3f})")

def visual_overlay(image, prob, fname):
    """image: (1,256,256), prob: (118,256,256)"""
    plt.figure(figsize=(12,4))
    plt.subplot(1, len(PLOT_CH)+1, 1)
    plt.imshow(image.squeeze(), cmap="gray"); plt.title("X‑ray"); plt.axis("off")
    for i, c in enumerate(PLOT_CH, start=2):
        plt.subplot(1, len(PLOT_CH)+1, i)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.imshow(prob[c], cmap="viridis", vmin=0, vmax=1, alpha=.4)
        plt.title(f"chan {c}"); plt.axis("off")
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

bce = nn.BCEWithLogitsLoss()

def combo_loss(logits, target):
    fl = focal_tversky_loss(logits, target,
                            alpha=.3, beta=.7, gamma=1.5,
                            ignore_empty=True)
    return 0.3 * bce(logits, target) + 0.7 * fl

def merge_prob_max(prob, classes):
    # pick the class subset → (B,len,K,H,W) then max over that dim
    merged = prob[:, classes, :, :].max(dim=1, keepdim=True).values
    return merged

# ---------------- main --------------------
def main():
    device = DEVICE
    print("device =", device)

    # datasets & loaders
    train_ds = XRaySegmentationDataset(TRAIN_IMG, TRAIN_MSK, transform=False)
    val_ds   = XRaySegmentationDataset(VAL_IMG,   VAL_MSK,   transform=False, eval=True)
    train_dl = DataLoader(train_ds, BATCH_TRAIN, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   BATCH_VAL,   shuffle=False,
                          num_workers=4, pin_memory=True)

    # model & optimiser
    # ckpt = torch.load("checkpoints/best_v3.pt", map_location=device, weights_only=False)
    model = MultiLabelSegModel().to(device)
    # model.load_state_dict(ckpt["model_state"])
    # epoch_start = ckpt["epoch"]
    # best_dice = ckpt["dice"]

    enc_params = list(model.encoder.parameters())
    enc_param_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_param_ids]
    BASE_LR = LR
    optim = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": 3e-4},
            {"params": dec_params, "lr": 3e-5},
        ],
        weight_decay=1e-4
    )
    # optim.load_state_dict(ckpt["optim_state"])
    scaler = GradScaler()

    # best_dice = ckpt["dice"]
    # train_loss = ckpt["train_loss"]
    # val_loss = ckpt["val_loss"]

    best_dice = 0
    train_loss = []
    val_loss = []

    for epoch in range(0, 100):
        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0

        for img, msk in train_dl:
            img, msk = img.to(device), msk.to(device)
            assert msk.sum() > 0, "train msk is all zeros"
            
            with torch.cuda.amp.autocast():
                logit = model(img)
                #print(f"Model output shape: {logit.shape}, Target shape: {msk.shape}")  # Debug print
                loss = combo_loss(logit, msk)
            

            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            optim.zero_grad()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_dl)

        # ---------- VALIDATE ----------
        model.eval(); macro_sum = 0; n = 0

        with torch.no_grad(), autocast():
            for img, msk in val_dl:
                img, msk = img.to(device), msk.to(device)
                logits = model(img.float())
                logits = torch.clamp(logits, -20, 20)
                prob = torch.sigmoid(logits)

                left_lung_prob = merge_prob_max(prob, [45, 48])
                right_lung_prob = merge_prob_max(prob, [46, 47, 49])
                heart_prob = merge_prob_max(prob, [28])

                left_lung_msk = msk[:, 1, :, :]
                right_lung_msk = msk[:, 0, :, :]
                heart_msk = msk[:, 2, :, :]


                prob = torch.cat((left_lung_prob, right_lung_prob, heart_prob), dim=1)
                msk = torch.stack((left_lung_msk, right_lung_msk, heart_msk), dim=1)

                macro_sum += macro_soft_dice(prob, msk); n += 1
            
            val_macro = macro_sum / n
                
                

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train‑loss {avg_loss:.4f} | val‑macroDice {val_macro:.3f}")
        train_loss.append(avg_loss)
        val_loss.append(val_macro)

        # checkpoint best
        if val_macro > best_dice:
            best_dice = val_macro
            save_best(model, optim, epoch, best_dice, train_loss, val_loss)

        # quick visual for one sample
        if epoch % 5 == 0 or epoch == 1:
            # pick first batch image
            img_cpu = img[0].cpu()
            prob_cpu = prob[0].cpu()
            visual_overlay(img_cpu, prob_cpu,
                           VIS_DIR / f"v2_epoch{epoch:02d}.png")


    print("Training done. Best val Dice =", best_dice)

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')

if __name__ == "__main__":
    main()
