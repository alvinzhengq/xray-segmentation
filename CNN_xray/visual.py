import torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from data import XRaySegmentationDataset
from model import MultiLabelSegModel
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast
# ---------------- CONFIG -----------------
DATA_ROOT   = Path("/home/user/yibo_workspace")          # adjust once
VAL_IMG     = DATA_ROOT/"images/visual"
VAL_MSK     = DATA_ROOT/"masks/visual"

BATCH_VAL   = 4
PLOT_CH     = [0, 1, 2]
VIS_DIR  = Path("runs/vis");   VIS_DIR.mkdir(exist_ok=True, parents=True)
def merge_prob_max(prob, classes):
    # pick the class subset → (B,len,K,H,W) then max over that dim
    merged = prob[:, classes, :, :].max(dim=1, keepdim=True).values
    return merged

def visual_overlay(image, prob, fname):
    """image: (1,256,256), prob: (3,256,256)"""
    plt.figure(figsize=(6, 6))
    plt.subplot(1, len(PLOT_CH)+1, 1)
    plt.imshow(image.squeeze(), cmap="gray"); plt.title("X‑ray GT"); plt.axis("off")
    for i, c in enumerate(PLOT_CH, start=1):
        plt.subplot(1, len(PLOT_CH)+1, i)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.imshow(prob[c], cmap="viridis", vmin=0, vmax=1, alpha=.4)
        plt.title(f"chan {c}"); plt.axis("off")
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

# ---------------- main --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # datasets & loaders
    val_ds = XRaySegmentationDataset(VAL_IMG, VAL_MSK, transform=False, eval=True)
    val_dl = DataLoader(val_ds, BATCH_VAL, shuffle=False,
                        num_workers=4, pin_memory=True)
    
    ckpt = torch.load("checkpoints/best_v2.pt", map_location=device, weights_only=False)
    model = MultiLabelSegModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad(), autocast():
        for img, msk in val_dl:
            img, msk = img.to(device), msk.to(device)
            # print(img.shape)
            # print(msk.shape)
            prob = torch.sigmoid(model(img))
            # print(prob.shape)

            left_prob = merge_prob_max(prob, [45, 48])
            right_prob = merge_prob_max(prob, [46, 47, 49])
            heart_prob = merge_prob_max(prob, [28])

            prob = torch.cat((right_prob, left_prob, heart_prob), dim=1)
            # print("prob shape: ", prob.shape)

            left_msk = msk[:, 1, :, :]
            right_msk = msk[:, 0, :, :]
            heart_msk = msk[:, 2, :, :]
            # print("msk shape: ", msk.shape)

            #img[0] is the left lung, img[1] is the right lung, img[2] is the heart
            img_cpu = img[0].cpu()
            prob_cpu = prob[0].cpu()
            msk_cpu = msk[0].cpu()
            # print("msk_cpu shape: ", msk_cpu.shape)
            visual_overlay(img_cpu, prob_cpu,
                           VIS_DIR / f"left_lung.png")
            visual_overlay(img_cpu, msk_cpu,
                           VIS_DIR / f"left_lung_gt.png")


            break

if __name__ == "__main__":
    main()


            
