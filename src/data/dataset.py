"""PyTorch dataset for 2D cardiac MRI segmentation."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nb
import numpy as np
from torchvision.transforms.functional import InterpolationMode
import torchvision


class CardiacDataset(Dataset):
    def __init__(self, patient_dirs, augmentation=False):
        super().__init__()
        self.patient_dirs = [Path(p) for p in patient_dirs]
        self.image_size = (256, 256)
        self.samples = []
        self.augmentation = augmentation

        images_path = []
        for patient_dir in self.patient_dirs:
            for nifti_path in patient_dir.glob('*.nii.gz'):
                if nifti_path.name.endswith('_gt.nii.gz'):
                    continue
                
                images_path.append(nifti_path)
        images_path = sorted(images_path)

        for image_path in images_path:
            mask_name = image_path.name.replace(".nii.gz", "_gt.nii.gz")
            mask_path = image_path.with_name(mask_name)

            if not mask_path.exists():
                raise FileNotFoundError

            nifti_image = nb.load(image_path)
            nb_coupes = nifti_image.shape[-1]

            indices = range(nb_coupes)

            for i in indices:
                tup = (image_path, mask_path, i)
                self.samples.append(tup)
    
    def __len__(self):
        return len(self.samples)

    def apply_augmentation(self, image, mask):
        height, width = image.shape[-2:]

        angle = torch.empty(1).uniform_(-10, 10).item()
        scale = torch.empty(1).uniform_(0.95, 1.05).item()

        max_translation_x = int(0.05 * width)
        max_translation_y = int(0.05 * height)

        translation_x = int(
            torch.randint(
                -max_translation_x,
                max_translation_x + 1,
                size=(1,),
            ).item()
        )

        translation_y = int(
            torch.randint(
                -max_translation_y,
                max_translation_y + 1,
                size=(1,),
            ).item()
        )

        image = torchvision.transforms.functional.affine(
            image,
            angle=angle,
            translate=[translation_x, translation_y],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

        mask = torchvision.transforms.functional.affine(
            mask,
            angle=angle,
            translate=[translation_x, translation_y],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )

        intensity_factor = torch.empty(1).uniform_(
            0.9,
            1.1,
        ).item()

        image = image * intensity_factor

        return image, mask

    def __getitem__(self, index):
        image_path, mask_path, indice  = self.samples[index]

        image_nifti = nb.load(image_path)
        mask_nifti = nb.load(mask_path)

        image_array = image_nifti.get_fdata(dtype=np.float32)
        mask_array = mask_nifti.dataobj

        # Récupération de la slice
        image_slice = image_array[:, :, indice]
        mask_slice = mask_array[:, :, indice]

        # Normalisation
        non_zero_mask = image_slice != 0.0
        foreground_pixels = image_slice[non_zero_mask]
        if foreground_pixels.size > 0:

            foreground_mean = foreground_pixels.mean()
            foreground_std = foreground_pixels.std()

            if foreground_std > 0:
                image_slice[non_zero_mask] = (foreground_pixels - foreground_mean) / foreground_std

        # Convertion en tenseur Pytorch
        image_tensor = torch.from_numpy(image_slice.copy()).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice.copy()).unsqueeze(0).long()

        # Resize
        image_resize = torchvision.transforms.functional.resize(image_tensor, 
                                                                self.image_size, 
                                                                interpolation=InterpolationMode.BILINEAR
                                                                )

        mask_resize = torchvision.transforms.functional.resize(mask_tensor, 
                                                               self.image_size, 
                                                               interpolation=InterpolationMode.NEAREST
                                                               )

        if self.augmentation:
            image_resize, mask_resize = self.apply_augmentation(
                image_resize,
                mask_resize,
            )
        mask_resize = mask_resize.squeeze(0)

        return(image_resize, mask_resize)


if __name__ == "__main__":

    patient_list = ["data/training/patient001","data/training/patient002"]

    dataset = CardiacDataset(patient_list)
    print(len(dataset))
    image, mask = dataset[10]
    print(f"Image type, shape: {image.dtype, image.shape}")
    print(f"Mask type, shape: {mask.dtype, mask.shape}")
    print(f"Classes: {np.unique(mask)}")
    print(f"Maximum, minimum: {image.max(), image.min()}")


         




