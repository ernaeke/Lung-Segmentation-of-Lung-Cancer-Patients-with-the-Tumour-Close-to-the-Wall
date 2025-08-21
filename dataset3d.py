import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt, gaussian_filter
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation, apply_orientation

class Lung3DDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=None, num_patches=4):
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')
        ])
        self.patch_size = patch_size
        self.num_patches = num_patches

    def _load_axial_volume(self, nii_path):
        nii = nib.load(nii_path)
        data = nii.get_fdata()
        current_ornt = io_orientation(nii.affine)
        target_ornt = axcodes2ornt(('R', 'A', 'S'))
        transform = ornt_transform(current_ornt, target_ornt)
        data = apply_orientation(data, transform)
        return data

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self._load_axial_volume(self.image_paths[idx])
        mask = self._load_axial_volume(self.mask_paths[idx])
        img = np.clip(img, 0, 1400) / 1400.0
        mask = (mask > 0).astype(np.float32)

        img, mask, tumor_mask = self._add_synthetic_tumor(img, mask)

        if self.patch_size is None:
            img = self._pad_to_multiple_of_16(img)
            mask = self._pad_to_multiple_of_16(mask)
            tumor_mask = self._pad_to_multiple_of_16(tumor_mask)
            return (
                torch.tensor(img[None], dtype=torch.float32),
                torch.tensor(mask[None], dtype=torch.float32),
                torch.tensor(tumor_mask[None], dtype=torch.float32)
            )

        while True:
            try:
                img_patches, mask_patches, tumor_patches = self._random_crop_patches(img, mask, tumor_mask)
                return (
                    torch.tensor(img_patches, dtype=torch.float32),
                    torch.tensor(mask_patches, dtype=torch.float32),
                    torch.tensor(tumor_patches, dtype=torch.float32)
                )
            except ValueError:
                idx = (idx + 1) % len(self.image_paths)

    def _random_crop_patches(self, img, mask, tumor_mask):
        D, H, W = img.shape
        d, h, w = self.patch_size

        if D < d or H < h or W < w:
            raise ValueError(f"❌ Volume too small: {img.shape} < patch size {self.patch_size}")

        img_patches, mask_patches, tumor_patches = [], [], []

        for _ in range(self.num_patches):
            z = np.random.randint(0, D - d + 1)
            y = np.random.randint(0, H - h + 1)
            x = np.random.randint(0, W - w + 1)

            img_patch = img[z:z+d, y:y+h, x:x+w]
            mask_patch = mask[z:z+d, y:y+h, x:x+w]
            tumor_patch = tumor_mask[z:z+d, y:y+h, x:x+w]

            img_patch = self._pad_to_multiple_of_16(img_patch)
            mask_patch = self._pad_to_multiple_of_16(mask_patch)
            tumor_patch = self._pad_to_multiple_of_16(tumor_patch)

            img_patches.append(img_patch[None])
            mask_patches.append(mask_patch[None])
            tumor_patches.append(tumor_patch[None])

        return (
            np.stack(img_patches),
            np.stack(mask_patches),
            np.stack(tumor_patches)
        )

    def _pad_to_multiple_of_16(self, volume):
        D, H, W = volume.shape
        pad_d = (16 - D % 16) % 16
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        return np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

    def _add_synthetic_tumor(self, img, mask, min_tumors=2, max_tumors=6):
        img_nii = nib.load(self.image_paths[0])
        spacing = img_nii.header.get_zooms()[:3]
        tumor_mask_total = np.zeros_like(mask, dtype=np.float32)
        img_augmented = img.copy()

        dist_inside = distance_transform_edt(mask, sampling=spacing)
        shell = ((dist_inside <= 5.0) & (dist_inside >= 1.0) & (mask > 0)).astype(np.uint8)
        shell_coords = np.argwhere(shell)
        if len(shell_coords) == 0:
            return img, mask, tumor_mask_total

        num_tumors_to_add = np.random.randint(min_tumors, max_tumors + 1)
        attempts, max_attempts, tumors_added = 0, 25, 0

        while tumors_added < num_tumors_to_add and attempts < max_attempts:
            attempts += 1
            center = shell_coords[np.random.randint(len(shell_coords))]
            # Smaller tumors with tighter distribution
            tumor_radius_mm = np.clip(np.random.normal(8.0, 3.5), 2.5, 15.0)
            tumor_radius_vox = np.round(np.array(tumor_radius_mm) / spacing).astype(int)
            margin = np.max(tumor_radius_vox)

            if any([
                center[0] < margin or center[0] >= img.shape[0] - margin,
                center[1] < margin or center[1] >= img.shape[1] - margin,
                center[2] < margin or center[2] >= img.shape[2] - margin
            ]):
                continue

            zz, yy, xx = np.meshgrid(
                np.arange(img.shape[0]),
                np.arange(img.shape[1]),
                np.arange(img.shape[2]),
                indexing='ij'
            )
            dist = (((zz - center[0]) / tumor_radius_vox[0]) ** 2 +
                    ((yy - center[1]) / tumor_radius_vox[1]) ** 2 +
                    ((xx - center[2]) / tumor_radius_vox[2]) ** 2)
            tumor_mask = (dist <= 1).astype(np.float32)

            overlap_ratio = (tumor_mask * shell).sum() / tumor_mask.sum()
            if overlap_ratio < 0.2:
                continue

            tumor_mask_blurred = gaussian_filter(tumor_mask, sigma=0.3)

            tumor_intensity = (np.random.uniform(900, 1200) if np.random.rand() < 0.5
                               else np.random.uniform(100, 300)) / 1400.0
            tumor = tumor_mask_blurred * tumor_intensity

            img_augmented = np.maximum(img_augmented, tumor)
            tumor_mask_total = np.maximum(tumor_mask_total, tumor_mask)
            tumors_added += 1

        mask = np.maximum(mask, tumor_mask_total)
        return img_augmented, mask, tumor_mask_total

