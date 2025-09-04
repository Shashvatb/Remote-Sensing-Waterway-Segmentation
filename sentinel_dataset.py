import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms

class SentinelWaterDataset(Dataset):
    def __init__(self, root_dir, tile_size=256, mode='train'):
        """
        init function for our dataset class
        we load the scenes, scale and clip the bands, calculate the (pseudo) label, crop the scene into managable tiles and oversample under represented class (water)
        root_dir: path to folder containing sentinel dataset with multiple scenes
        tile_size: size of the square tiles to crop from the images
        """
        self.root_dir = root_dir
        self.tile_size = tile_size

        # Load all folder names
        self.scenes = os.listdir(self.root_dir)
        if mode == 'train':
            self.scenes = self.scenes[:-1]
        else:
            self.scenes = self.scenes[-1]

        # Preprocess all scenes by extracting bands and computing NDWI
        self.tiles = []
        self.masks = []
        for scene in self.scenes:
            band_paths = self.get_band_paths(scene)
            img, ndwi = self.load_and_preprocess(band_paths)
            # Convert to smaller tiles
            scene_tiles, scene_masks = self.tile_image(img, ndwi)
            self.tiles.extend(scene_tiles)
            self.masks.extend(scene_masks)
        
        # Oversample rare class tiles (tiles with water pixels < threshold)
        self.balance_tiles()

        # Transform to torch tensors
        self.to_tensor = transforms.ToTensor()

    def get_band_paths(self, scene):
        """
        Extract file paths for B02(Blue), B03(Green), B04(Red), B08(NIR)
        """
        img_data_path = os.path.join(scene, "GRANULE")
        granules = os.listdir(img_data_path)
        granule_path = os.path.join(img_data_path, granules[0], "IMG_DATA","R10m")
        bands = {}
        for f in os.listdir(granule_path):
            if f.endswith('.jp2'):
                if "_B02" in f:
                    bands['B02'] = os.path.join(granule_path, f)
                elif "_B03" in f:
                    bands['B03'] = os.path.join(granule_path, f)
                elif "_B04" in f:
                    bands['B04'] = os.path.join(granule_path, f)
                elif "_B08" in f:
                    bands['B08'] = os.path.join(granule_path, f)
        return bands
    
    def load_band(self, path):
        """
        Load the bands into memory and scale and clip them
        """    
        with rasterio.open(path) as src:
            band = src.read(1).astype('float32') / 10000
            band = np.clip(band, 0, 1)
        return band

    def load_and_preprocess(self, band_paths):
        """
        Load the bands, normalize, compute NDWI
        """
        blue = self.load_band(band_paths['B02'])
        green = self.load_band(band_paths['B03'])
        red = self.load_band(band_paths['B04'])
        nir = self.load_band(band_paths['B08'])

        # Stack bands (C, H, W)
        img = np.stack([blue, green, red, nir], axis=0)

        # Compute NDWI
        ndwi = (green - nir) / (green + nir)
        ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)
        # Convert to binary mask: water if NDWI > 0
        mask = (ndwi > 0.0).astype(np.uint8)
        return img, mask

    def tile_image(self, img, mask):
        """
        Slice the image and mask into smaller tiles
        """
        C, H, W = img.shape
        tiles = []
        masks = []
        for i in range(0, H, self.tile_size):
            for j in range(0, W, self.tile_size):
                tile = img[:, i:i+self.tile_size, j:j+self.tile_size]
                tile_mask = mask[i:i+self.tile_size, j:j+self.tile_size]
                if tile.shape[1] == self.tile_size and tile.shape[2] == self.tile_size:
                    tiles.append(tile)
                    masks.append(tile_mask)
        return tiles, masks

    def balance_tiles(self):
        """
        Oversample tiles with water pixels using flips and rotations
        """
        balanced_tiles = []
        balanced_masks = []

        for tile, mask in zip(self.tiles, self.masks):
            balanced_tiles.append(tile)
            balanced_masks.append(mask)
            # If water pixels exist, apply augmentation
            if mask.sum() > 0:
                # Horizontal flip
                if random.random() > 0.25:
                    balanced_tiles.append(np.flip(tile, axis=2))
                    balanced_masks.append(np.flip(mask, axis=1))
                # Vertical flip
                if random.random() > 0.25:
                    balanced_tiles.append(np.flip(tile, axis=1))
                    balanced_masks.append(np.flip(mask, axis=0))
                # 90-degree rotation
                if random.random() > 0.25:
                    balanced_tiles.append(np.rot90(tile, k=1, axes=(1,2)))
                    balanced_masks.append(np.rot90(mask, k=1))

        self.tiles = balanced_tiles
        self.masks = balanced_masks

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        x = self.tiles[idx]
        y = self.masks[idx]
        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
