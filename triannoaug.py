import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from unet3d import UNet3D
from datasetnoaug import Lung3DDataset
from losses import dice_loss_3d
from scipy.ndimage import binary_fill_holes

# ---- Config ----
image_dir = "Path_to_scans"
mask_dir = "Path_to_masks"
batch_size = 1
epochs = 25
lr = 1e-4
patch_size = None
num_patches = 1

# ---- Dataset & Loader ----
dataset = Lung3DDataset(image_dir, mask_dir, patch_size=patch_size, num_patches=num_patches)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---- Model ----
torch.cuda.empty_cache()
device = torch.device("cuda:0")
model = UNet3D()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses, val_losses = [], []

def save_volume_slices_as_jpg(image_volume, mask_volume, output_dir, case_name):
    """
    Draws a solid green overlay (with alpha) wherever mask>0.
    """
    case_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)

    for z in range(image_volume.shape[2]):
        img_slice = image_volume[:, :, z]
        mask_slice = mask_volume[:, :, z]

        # Rotate for landscape preview
        img_rot = np.rot90(img_slice)
        mask_rot = np.rot90(mask_slice)

        mask_bin = (mask_rot > 0).astype(np.float32)

        plt.figure(figsize=(8, 4.5))
        plt.imshow(img_rot, cmap='gray', vmin=0, vmax=1)

        # GREEN overlay (0,255,0) with alpha
        overlay = np.zeros((*mask_bin.shape, 4), dtype=np.float32)
        overlay[..., 1] = 1.0            # green channel
        overlay[..., 3] = mask_bin * 0.4 # alpha
        plt.imshow(overlay, interpolation='none')

        plt.axis('off')
        plt.title(f"{case_name} - Slice {z}")
        plt.savefig(os.path.join(case_dir, f"{case_name}_slice_{z:03d}.jpg"),
                    bbox_inches='tight', pad_inches=0)
        plt.close()


# ---- Training ----
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for img_batch, mask_batch in train_loader:
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)

        pred = model(img_batch)
        loss = dice_loss_3d(pred, mask_batch)

        print(f"[Train Epoch {epoch+1}] Pred stats: "
              f"max={pred.max().item():.4f}, min={pred.min().item():.4f}, "
              f"mean={pred.mean().item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        torch.cuda.empty_cache()

    train_losses.append(total_loss / len(train_loader))

    # ---- Validation ----
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for val_img, val_mask in val_loader:
            val_img = val_img.to(device)
            val_mask = val_mask.to(device)
            pred_val = model(val_img)
            loss_val = dice_loss_3d(pred_val, val_mask)
            val_total += loss_val.item()

            print(f"[Val Epoch {epoch+1}] Pred stats: "
                  f"max={pred_val.max().item():.4f}, min={pred_val.min().item():.4f}, "
                  f"mean={pred_val.mean().item():.4f}")

            torch.cuda.empty_cache()

    val_losses.append(val_total / len(val_loader))
    print(f"Epoch {epoch+1} ✅ Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")

    torch.save(model.state_dict(), f"noaug_unet3d_epoch{epoch+1}.pth")

    # ---- Save validation overlays (no augmentation) ----
    val_img_np = val_img.detach().cpu().numpy()[0, 0]
    pred_mask_np = (pred_val.detach().cpu().numpy()[0, 0] > 0.5).astype(np.uint8)

    # Optional: slice-wise hole fill for cleaner visuals
    pred_mask_np_filled = np.zeros_like(pred_mask_np)
    for z in range(pred_mask_np.shape[2]):
        pred_mask_np_filled[:, :, z] = binary_fill_holes(pred_mask_np[:, :, z]).astype(np.uint8)

    gt_mask_np = val_mask.detach().cpu().numpy()[0, 0]

    save_volume_slices_as_jpg(val_img_np, pred_mask_np_filled,
                              output_dir="fixednoaug_jpg_output_pred25",
                              case_name=f"epoch_{epoch+1}")
    save_volume_slices_as_jpg(val_img_np, gt_mask_np,
                              output_dir="fixednoaug_jpg_output_gt25",
                              case_name=f"epoch_{epoch+1}")

# ---- Loss Plot ----
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Dice Loss")
plt.title("Training and Validation Dice Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot_3d_noaug.png")
plt.show()

