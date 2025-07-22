import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import logging
import numpy as np

# Disable PIL warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.images = []
        self.labels = []
        
        # Enhanced image loading with quality filtering
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        for label in ['real', 'fake']:
            label_dir = os.path.join(data_dir, label)
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith(valid_extensions):
                        img_path = os.path.join(label_dir, img_name)
                        # Quick quality check
                        try:
                            with Image.open(img_path) as img:
                                img = img.convert('RGB')
                                if img.size[0] >= 50 and img.size[1] >= 50:  # Min size filter
                                    self.images.append(img_path)
                                    self.labels.append(1 if label == 'real' else 0)
                        except Exception:
                            continue  # Skip corrupted images silently
        
        if len(self.images) == 0:
            print(f'No valid images found in {data_dir}')
        else:
            print(f'Loaded {len(self.images)} images')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Use a more robust approach to handle corrupted images
        attempts = 0
        max_attempts = 5
        
        while attempts < max_attempts:
            try:
                img_path = self.images[idx % len(self.images)]
                
                # Load and validate image
                with Image.open(img_path) as img:
                    image = img.convert('RGB')
                    
                    # Check image is valid
                    if image.size[0] < 32 or image.size[1] < 32:
                        raise ValueError("Image too small")
                    
                    # Apply transforms
                    if self.transform:
                        image = self.transform(image)
                    
                    label = self.labels[idx % len(self.labels)]
                    return image, label
                    
            except Exception:
                # Move to next image silently
                idx = (idx + 1) % len(self.images)
                attempts += 1
        
        # If all attempts fail, return a dummy tensor
        if self.transform:
            dummy_image = self.transform(Image.new('RGB', (224, 224), color='black'))
        else:
            dummy_image = transforms.ToTensor()(Image.new('RGB', (224, 224), color='black'))
        
        return dummy_image, 0 