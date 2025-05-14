import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import logging

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
        
        # Đọc ảnh từ thư mục train/real và train/fake
        for label in ['real', 'fake']:
            label_dir = os.path.join(data_dir, label)
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(label_dir, img_name))
                        self.labels.append(1 if label == 'real' else 0)
        
        if len(self.images) == 0:
            logging.warning(f'Không tìm thấy ảnh trong thư mục {data_dir}')
        else:
            logging.info(f'Tổng số ảnh: {len(self.images)}')
            logging.info(f'Số ảnh thật: {sum(self.labels)}')
            logging.info(f'Số ảnh giả: {len(self.labels) - sum(self.labels)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            logging.error(f'Lỗi khi đọc ảnh {img_path}: {str(e)}')
            return None 