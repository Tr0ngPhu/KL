# Standard library imports
import os
import warnings
import random

# Third-party imports
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Disable PIL warnings
warnings.filterwarnings("ignore", category=UserWarning)

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, use_albumentations=False):
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.images = []
        self.labels = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        class_counts = {'real': 0, 'fake': 0}

        # Try the provided data_dir, then fallback to src/data/... if empty
        tried_dirs = [data_dir]
        if os.path.normpath(data_dir).startswith('data') and not os.path.isabs(data_dir):
            alt_dir = os.path.join('src', data_dir)
            tried_dirs.append(alt_dir)
        elif os.path.normpath(data_dir).startswith(os.path.join(os.getcwd(), 'data')):
            alt_dir = os.path.join(os.getcwd(), 'src', 'data', os.path.basename(data_dir))
            tried_dirs.append(alt_dir)

        found = False
        for try_dir in tried_dirs:
            self.images.clear()
            self.labels.clear()
            class_counts = {'real': 0, 'fake': 0}
            for label in ['real', 'fake']:
                label_dir = os.path.join(try_dir, label)
                if not os.path.exists(label_dir) or not os.path.isdir(label_dir):
                    continue
                for img_name in os.listdir(label_dir):
                    # Only accept files with valid image extensions, skip hidden and non-image files
                    if not img_name.lower().endswith(valid_extensions):
                        continue
                    img_path = os.path.join(label_dir, img_name)
                    # Skip files that are not actual files (e.g., directories, .gitkeep, etc.)
                    if not os.path.isfile(img_path):
                        continue
                    try:
                        with Image.open(img_path) as img:
                            img = img.convert('RGB')
                            if img.size[0] >= 50 and img.size[1] >= 50:
                                self.images.append(img_path)
                                self.labels.append(1 if label == 'real' else 0)
                                class_counts[label] += 1
                    except Exception:
                        pass
            if len(self.images) > 0:
                found = True
                break
        if not found:
            raise RuntimeError('No valid images found in any of the tried directories. Please check your data folders.')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        with Image.open(img_path) as img:
            image = img.convert('RGB')
            if image.size[0] < 32 or image.size[1] < 32:
                raise ValueError("Image too small")
            if self.transform:
                image = self.transform(image)
            return image, label