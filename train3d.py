import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from unet3d import UNet3D
from dataset3d import Lung3DDataset
from losses import dice_loss_3d
import numpy.ma as ma
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_fill_holes

# 🔁 Clear any previous cache and set fragmentation handling
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---- Config ----
image_dir = "Path_to_scans"
mask_dir = "Path_to_masks"
batch_size = 1
epochs = 25
lr = 1e-4
patch_size = None  # Set to ((64, 128, 128)) if you want patch-based
num_patches = 1

dataset = Lung3DDataset(image_dir, mask_dir, patch_size=patch_size, num_patches=num_patches)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 🔁 Clear again before model allocation
torch.cuda.empty_cache()

model = UNet3D().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
val_losses = []
focused_losses = []

# ---- Save slices ----
def save_volume_slices_as_jpg(image_volume, mask_volume, output_dir, case_name):
    case_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)
    D = image_volume.shape[2]
    for z in range(D):
        img_slice = image_volume[:, :, z]
        mask_slice = mask_volume[:, :, z]
        img_rot = np.rot90(img_slice, k=1)
        mask_rot = np.rot90(mask_slice, k=1)

        print(f"[DEBUG] {case_name} - Slice {z} max mask val: {mask_rot.max()}")

        plt.figure(figsize=(8, 4.5))
        plt.imshow(img_rot, cmap='gray')
        if mask_rot.max() > 0:
            plt.imshow(ma.masked_where(mask_rot == 0, mask_rot), cmap='Greens', vmin=0, vmax=1, alpha=0.4)
        else:
            print(f"⚠️ Mask missing in slice {z}")
        plt.axis('off')
        plt.title(f"{case_name} - Slice {z}")
        plt.savefig(os.path.join(case_dir, f"{case_name}_slice_{z:03d}.jpg"), bbox_inches='tight', pad_inches=0)
        plt.close()


# ---- Tumor-focused mask ----
def get_focus_mask(tumor_mask, spacing, margin_mm=20):
    dist = distance_transform_edt(1 - tumor_mask, sampling=spacing)
    return (dist <= margin_mm).astype(np.float32)

# ---- Training ----
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for img_batch, mask_batch, tumor_mask_batch in train_loader:
        img_batch = img_batch.cuda()
        mask_batch = mask_batch.cuda()
        tumor_mask_batch = tumor_mask_batch.cuda()

        pred = model(img_batch)
        # Main dice loss (full volume)
        loss_main = dice_loss_3d(pred, mask_batch)

# Tumor-focused loss (optional emphasis)
        spacing = (2.5, 1.0, 1.0)
        focus_mask = torch.tensor(
            get_focus_mask(tumor_mask_batch.cpu().numpy()[0, 0], spacing),
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).cuda()

        pred_focus = pred * focus_mask
        mask_focus = mask_batch * focus_mask
        loss_focus = dice_loss_3d(pred_focus, mask_focus)

# Total loss: weight focus higher
        loss = loss_main + 0.5 * loss_focus  # You can try 0.5 or 1.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 🔁 Optional: clear cache per batch
        torch.cuda.empty_cache()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---- Validation ----
    model.eval()
    val_total = 0.0
    focus_total = 0.0

    with torch.no_grad():
        for val_img, val_mask, tumor_mask in val_loader:
            val_img = val_img.cuda()
            val_mask = val_mask.cuda()
            tumor_mask = tumor_mask.cuda()

            pred_val = model(val_img)
            loss_val = dice_loss_3d(pred_val, val_mask)
            val_total += loss_val.item()

            # ✅ Focused loss around tumor (+/- 2cm)
            spacing = (2.5, 1.0, 1.0)
            focus_mask = torch.tensor(
                get_focus_mask(tumor_mask.cpu().numpy()[0, 0], spacing),
                dtype=torch.float32
            ).cuda()
            pred_focus = pred_val * focus_mask
            mask_focus = val_mask * focus_mask
            loss_focus = dice_loss_3d(pred_focus, mask_focus)
            focus_total += loss_focus.item()

            # 🔁 Clear cache per validation step
            torch.cuda.empty_cache()

    avg_val_loss = val_total / len(val_loader)
    avg_focus_loss = focus_total / len(val_loader)
    val_losses.append(avg_val_loss)
    focused_losses.append(avg_focus_loss)

    print(f"Epoch {epoch+1} ✅ Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Focused (±2cm): {avg_focus_loss:.4f}")

    torch.save(model.state_dict(), f"unet3d_epoch{epoch+1}.pth")

    val_img_np = val_img.detach().cpu().numpy()[0, 0]
    pred_mask_np = (pred_val.detach().cpu().numpy()[0, 0] > 0.5).astype(np.uint8)

# ✅ Fill holes in the prediction mask slice-by-slice
    pred_mask_np_filled = np.zeros_like(pred_mask_np)
    for z in range(pred_mask_np.shape[2]):
        pred_mask_np_filled[:, :, z] = binary_fill_holes(pred_mask_np[:, :, z]).astype(np.uint8)

    gt_mask_np = val_mask.detach().cpu().numpy()[0, 0]

    save_volume_slices_as_jpg(val_img_np, pred_mask_np, output_dir="25epochs_jpg_output_pred2", case_name=f"epoch_{epoch+1}")
    save_volume_slices_as_jpg(val_img_np, gt_mask_np, output_dir="25epochs_jpg_output_gt2", case_name=f"epoch_{epoch+1}")
    # 🧠 Visualize synthetic tumor placement (optional)
    tumor_mask_np = tumor_mask.detach().cpu().numpy()[0, 0]
    save_volume_slices_as_jpg(val_img_np, tumor_mask_np, output_dir="jpg_output_tumors2", case_name=f"epoch_{epoch+1}")

    # 🔁 End-of-epoch cleanup
    torch.cuda.empty_cache()

# ---- Plot ----
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
plt.plot(range(1, epochs + 1), focused_losses, label="Focused (±2cm) Loss")
plt.xlabel("Epoch")
plt.ylabel("Dice Loss")
plt.title("Training Losses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("2loss_plot_3d_with_focus.png")
plt.show()

