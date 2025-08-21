import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import csv
from unet3d import UNet3D

def save_slice_jpg(img_slice, pred_mask_slice, gt_mask_slice, case_dir, z):
    """Overlay prediction (green) only."""
    plt.figure(figsize=(8, 4))  # Landscape

    img_slice = np.rot90(img_slice)
    pred_mask_slice = np.rot90(pred_mask_slice)
    gt_mask_slice = np.rot90(gt_mask_slice)  # kept for identical signature/flow

    plt.imshow(img_slice, cmap='gray')

    # Overlay prediction in green (only)
    green_mask = np.zeros((*pred_mask_slice.shape, 4))
    green_mask[..., 1] = 1.0  # Green channel
    green_mask[..., 3] = pred_mask_slice * 0.4
    plt.imshow(green_mask)

    # (GT and overlap overlays removed)

    plt.axis('off')
    plt.title(f"Slice {z}")
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, f"slice_{z:03d}.jpg"), bbox_inches='tight', pad_inches=0)
    plt.close()

def dice_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)

def generate_mask_3d(model, scan_path, output_root, mask_dir, csv_writer):
    case_name = os.path.splitext(os.path.basename(scan_path))[0]
    img_nii = nib.load(scan_path)
    image_np = img_nii.get_fdata()
    image_np = np.clip(image_np, 0, 1400) / 1400.0

    # Load ground truth mask
    mask_path = os.path.join(mask_dir, f"{case_name}.nii.gz")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(mask_dir, f"{case_name}.nii")

    if not os.path.exists(mask_path):
        print(f"⚠️ Ground truth mask not found for {scan_path}")
        csv_writer.writerow([case_name, "", "no"])
        return

    gt_mask_np = nib.load(mask_path).get_fdata()
    gt_mask_np = (gt_mask_np > 0).astype(np.uint8)

    # Prepare input tensor
    image_tensor = torch.tensor(image_np.transpose(2, 1, 0)[None, None], dtype=torch.float32).cuda()

    with torch.no_grad():
        pred = model(image_tensor)
        pred_soft = pred.squeeze().cpu().numpy()
        pred_mask = (pred_soft > 0.2).astype(np.uint8)

    pred_mask = pred_mask.transpose(2, 1, 0)
    pred_soft = pred_soft.transpose(2, 1, 0)

    # Match shapes
    min_z = min(pred_mask.shape[2], image_np.shape[2], gt_mask_np.shape[2])
    image_np = image_np[:, :, :min_z]
    pred_mask = pred_mask[:, :, :min_z]
    pred_soft = pred_soft[:, :, :min_z]
    gt_mask_np = gt_mask_np[:, :, :min_z]

    # Compute Dice
    dice = dice_score(pred_mask, gt_mask_np)
    csv_writer.writerow([case_name, f"{dice:.4f}", "yes"])

    # Save slices
    case_dir = os.path.join(output_root, case_name)
    os.makedirs(case_dir, exist_ok=True)

    for z in range(pred_mask.shape[2]):
        img_slice = image_np[:, :, z]
        pred_slice = pred_mask[:, :, z]
        gt_slice = gt_mask_np[:, :, z]
        save_slice_jpg(img_slice, pred_slice, gt_slice, case_dir, z)
        print(f"✅ Saved: {case_name}/slice_{z:03d}.jpg")

    # Save volumes
    nib.save(nib.Nifti1Image(pred_mask, img_nii.affine), os.path.join(case_dir, f"{case_name}_pred.nii.gz"))
    nib.save(nib.Nifti1Image(image_np, img_nii.affine), os.path.join(case_dir, f"{case_name}_input.nii.gz"))
    nib.save(nib.Nifti1Image(pred_soft, img_nii.affine), os.path.join(case_dir, f"{case_name}_probs.nii.gz"))

if __name__ == "__main__":
    model = UNet3D().cuda()
    model.load_state_dict(torch.load("/home/erna/lung_seg/unet3d_epoch25.pth"))
    model.eval()

    input_dir = "Path_to_scans"
    mask_dir = "Path_to_masks"
    output_root = "3Fixedwithaug25epochs"
    os.makedirs(output_root, exist_ok=True)

    dice_csv_path = os.path.join(output_root, "dice_scores.csv")
    with open(dice_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Case", "Dice", "GT_Found"])

        for fname in sorted(os.listdir(input_dir)):
            if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                scan_path = os.path.join(input_dir, fname)
                generate_mask_3d(model, scan_path, output_root, mask_dir, writer)
