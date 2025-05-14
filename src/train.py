# KL/train.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import yaml

# Thêm đường dẫn gốc vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.model import VisionTransformer
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils.utils import ensure_dir, check_admin_access, setup_logging, save_checkpoint
from data.dataset import CustomDataset

# Thiết lập logging
os.makedirs('../logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('../logs', 'training.log')),
        logging.StreamHandler()
    ]
)

class ProductDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.images = []
        self.labels = []
        self.class_names = ['Giả', 'Thật']
        
        # Đọc ảnh từ thư mục train
        train_dir = os.path.join(data_dir, 'train')
        real_dir = os.path.join(train_dir, 'real')
        fake_dir = os.path.join(train_dir, 'fake')
        
        # Đọc ảnh thật
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(real_dir, img_name))
                    self.labels.append(1)  # 1 cho hàng thật
        
        # Đọc ảnh giả
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(fake_dir, img_name))
                    self.labels.append(0)  # 0 cho hàng giả
        
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

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='min', verbose=True):
        """
        Args:
            patience (int): Số epoch chờ đợi trước khi dừng
            min_delta (float): Ngưỡng cải thiện tối thiểu
            mode (str): 'min' cho loss, 'max' cho accuracy
            verbose (bool): In thông báo chi tiết
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, value, epoch):
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
        elif self.mode == 'min':
            if value > self.best_value - self.min_delta:
                self.counter += 1
                if self.verbose:
                    logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        logging.info(f'Early stopping triggered. Best value: {self.best_value:.4f} at epoch {self.best_epoch}')
            else:
                self.best_value = value
                self.best_epoch = epoch
                self.counter = 0
        else:  # mode == 'max'
            if value < self.best_value + self.min_delta:
                self.counter += 1
                if self.verbose:
                    logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        logging.info(f'Early stopping triggered. Best value: {self.best_value:.4f} at epoch {self.best_epoch}')
            else:
                self.best_value = value
                self.best_epoch = epoch
                self.counter = 0

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    """Huấn luyện model với các tham số đã cho"""
    # Tạo thư mục lưu kết quả
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Khởi tạo các biến theo dõi
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=15, mode='max')
    
    # Huấn luyện
    for epoch in range(num_epochs):
        # Chế độ huấn luyện
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Huấn luyện trên tập train
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Xóa gradient cũ
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass và tối ưu
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Cập nhật thống kê
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Cập nhật progress bar
            train_bar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Tính toán metrics cho epoch
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Chế độ đánh giá
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        # Đánh giá trên tập validation
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Cập nhật thống kê
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Lưu predictions và labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Cập nhật progress bar
                val_bar.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Tính toán metrics cho epoch
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Cập nhật learning rate
        scheduler.step(val_loss)
        
        # Lưu metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # In thông tin epoch
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Lưu model tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, os.path.join(results_dir, 'best_model.pth'))
            logging.info(f'Model tốt nhất được lưu với accuracy: {best_val_acc:.2f}%')
        
        # Early stopping
        early_stopping(val_acc, epoch)
        if early_stopping.early_stop:
            logging.info('Early stopping triggered')
            break
    
    # Vẽ đồ thị và lưu kết quả
    plot_training_history(train_losses, val_losses, train_accs, val_accs, results_dir)
    plot_confusion_matrix(all_labels, all_preds, ['Giả', 'Thật'], results_dir)
    
    # In báo cáo phân loại
    report = classification_report(all_labels, all_preds, target_names=['Giả', 'Thật'])
    logging.info('\nClassification Report:\n' + report)
    
    return model, best_val_acc

def get_weighted_sampler(labels):
    """Tạo weighted sampler để xử lý mất cân bằng dữ liệu"""
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

def load_config():
    config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config()
    
    # Kiểm tra GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Tạo transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Tạo dataset và dataloader
    train_dataset = CustomDataset(
        data_dir=os.path.join(current_dir, 'data', 'train'),
        transform=train_transform
    )
    
    val_dataset = CustomDataset(
        data_dir=os.path.join(current_dir, 'data', 'validation'),
        transform=val_transform
    )
    
    # Tạo weighted sampler cho tập train
    train_sampler = get_weighted_sampler(train_dataset.labels)
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Khởi tạo model
    model = VisionTransformer(
        image_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        num_classes=config['model']['num_classes'],
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        mlp_dim=config['model']['mlp_dim']
    ).to(device)
    
    # Loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )
    
    # Huấn luyện model
    model, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=config['training']['epochs'], device=device
    )
    
    logging.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    main()