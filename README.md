# Hyperspectral augmentations

Implementation of transformations for hyperspectral images using PyTorch modules. This repo was created to use hyperspectral transformations with [BYOL](https://github.com/lucidrains/byol-pytorch).

Below is a modified version of `byol-pytorch/byol_pytorch/byol_pytorch.py` with the hyperspectral transformations.
```
import hyperspectralTransforms as HST

DEFAULT_AUG = torch.nn.Sequential(
                RandomApply( 
                HST.ColorJitter(0.8, 0.8, 0.8, 0.3), 
                p = 0.3 
            ),
            HST.RandomGrayscale(p=0.3, channels=12),
            HST.RandomHorizontalFlip(),
            RandomApply(
                filters.GaussianBlur2d((3, 3), (1.0, 2.0)), 
                p=0.2
            ),
            HST.RandomResizedCrop((image_size, image_size), scale=(0.5, 1.0))
	    )
```
