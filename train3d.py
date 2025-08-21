import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
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

        if self.patch_size is None:
            img = self._pad_to_multiple_of_16(img)
            mask = self._pad_to_multiple_of_16(mask)
            return (
                torch.tensor(img[None], dtype=torch.float32),
                torch.tensor(mask[None], dtype=torch.float32)
            )

        while True:
            try:
                img_patches, mask_patches = self._random_crop_patches(img, mask)
                return (
                    torch.tensor(img_patches, dtype=torch.float32),
                    torch.tensor(mask_patches, dtype=torch.float32)
                )
            except ValueError:
                idx = (idx + 1) % len(self.image_paths)

    def _random_crop_patches(self, img, mask):
        D, H, W = img.shape
        d, h, w = self.patch_size

        if D < d or H < h or W < w:
            raise ValueError(f"❌ Volume too small: {img.shape} < patch size {self.patch_size}")

        img_patches, mask_patches = [], []

        for _ in range(self.num_patches):
            z = np.random.randint(0, D - d + 1)
            y = np.random.randint(0, H - h + 1)
            x = np.random.randint(0, W - w + 1)

            img_patch = img[z:z+d, y:y+h, x:x+w]
            mask_patch = mask[z:z+d, y:y+h, x:x+w]

            img_patch = self._pad_to_multiple_of_16(img_patch)
            mask_patch = self._pad_to_multiple_of_16(mask_patch)

            img_patches.append(img_patch[None])
            mask_patches.append(mask_patch[None])

        return (
            np.stack(img_patches),
            np.stack(mask_patches)
        )

    def _pad_to_multiple_of_16(self, volume):
        D, H, W = volume.shape
        pad_d = (16 - D % 16) % 16
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        return np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

