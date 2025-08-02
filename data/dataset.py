import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging
import random

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
        
        for label in ['Real', 'Fake']:
            label_dir = os.path.join(data_dir, label)
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(label_dir, img_name)
                        if self._is_valid_image(img_path):
                            self.images.append(img_path)
                            self.labels.append(1 if label.lower() == 'real' else 0)
                        else:
                            logging.warning(f'Bỏ qua ảnh lỗi: {img_path}')
        
        if len(self.images) == 0:
            logging.warning(f'Không tìm thấy ảnh trong thư mục {data_dir}')
        else:
            logging.info(f'Tổng số ảnh hợp lệ: {len(self.images)}')
            logging.info(f'Số ảnh thật: {sum(self.labels)}')
            logging.info(f'Số ảnh giả: {len(self.labels) - sum(self.labels)}')
        
        self.labels = torch.tensor(self.labels)
    
    def _is_valid_image(self, img_path):
        """Kiểm tra xem ảnh có thể mở và xử lý được không"""
        try:
            with Image.open(img_path) as img:
                img.verify()
            img = Image.open(img_path).convert('RGB')
            if img.size[0] <= 0 or img.size[1] <= 0:
                return False
            return True
        except Exception as e:
            logging.error(f'Ảnh không hợp lệ {img_path}: {str(e)}')
            return False
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        max_retries = 10
        retries = 0
        
        while retries < max_retries:
            try:
                img_path = self.images[idx]
                label = self.labels[idx]
                
                image = Image.open(img_path).convert('RGB')
                logging.debug(f"Đã mở ảnh {img_path}, kích thước: {image.size}")
                
                if image.size[0] == 0 or image.size[1] == 0:
                    raise ValueError(f"Ảnh có kích thước không hợp lệ: {image.size}")
                
                if self.transform:
                    image = self.transform(image)
                    logging.debug(f"Đã áp dụng transform cho {img_path}, tensor shape: {image.shape}")
                
                return image, label
                
            except Exception as e:
                logging.error(f'Lỗi khi đọc ảnh {img_path}: {str(e)}')
                retries += 1
                idx = random.randint(0, len(self.images) - 1)
        
        raise RuntimeError(f'Không thể tìm thấy ảnh hợp lệ sau {max_retries} lần thử')

def debug_image_info(img_path):
    try:
        with Image.open(img_path) as img:
            print(f"File: {img_path}")
            print(f"Format: {img.format}")
            print(f"Mode: {img.mode}")
            print(f"Size: {img.size}")
            print(f"Info: {img.info}")
    except Exception as e:
        print(f"Error reading {img_path}: {e}")

def test_transforms(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(img_path).convert('RGB')
        print(f"Original image size: {image.size}")
        transformed = transform(image)
        print(f"Transformed tensor shape: {transformed.shape}")
        return transformed
    except Exception as e:
        print(f"Transform error: {e}")
        return None