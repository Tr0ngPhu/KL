# KL/train.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import yaml
import signal   
import psutil
# Thêm đường dẫn gốc vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from model import VisionTransformer
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils import ensure_dir, check_admin_access, setup_logging, save_checkpoint
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

def setup_interrupt_handlers(model, results_dir):
    def handle_interrupt(signum, frame):
        print("\nNhận tín hiệu dừng, đang lưu model...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'interrupted': True
        }, os.path.join(results_dir, 'interrupted_model.pth'))
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)


def log_system_stats():
    mem = psutil.virtual_memory()
    stats = {
        'cpu_usage': psutil.cpu_percent(),
        'memory_used_GB': mem.used / (1024**3),
        'memory_available_GB': mem.available / (1024**3)
    }
    logging.info(f"System Stats - CPU: {stats['cpu_usage']}% | "
                f"Mem Used: {stats['memory_used_GB']:.2f}GB | "
                f"Available: {stats['memory_available_GB']:.2f}GB")

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
    
    setup_interrupt_handlers(model, results_dir)

    # Khởi tạo các biến theo dõi
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=15, mode='max')
    
    # Huấn luyện
    for epoch in range(num_epochs):
        # Log thông tin hệ thống
        log_system_stats()
        # Giải phóng bộ nhớ GPU
        if device.type == 'mps':
            torch.mps.empty_cache()

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
                inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
                
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

def get_weighted_sampler(labels, beta=0.999):
    """Tạo weighted sampler với class balancing mạnh hơn
    
    Args:
        labels: Mảng numpy chứa nhãn (0=real, 1=fake)
        beta: Hyperparameter điều chỉnh độ mạnh của balancing (0.9-0.999)
    """
    class_counts = np.bincount(labels)
    
    # Effective number of samples (thay vì nghịch đảo đơn thuần)
    effective_num = 1.0 - np.power(beta, class_counts)
    class_weights = (1.0 - beta) / (effective_num + 1e-6)  # Tránh chia 0
    
    # Chuẩn hóa weights
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def load_config():
    config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_device():
    if torch.backends.mps.is_available():
        try:
            torch.mps.set_per_process_memory_fraction(0.75),# Giới hạn 90% bộ nhớ
            torch.mps.empty_cache()  # Giải phóng bộ nhớ  
            return torch.device("mps")
        except Exception as e:
            logging.warning(f"Không thể thiết lập bộ nhớ MPS: {str(e)}")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    


def main():
    # Load config
    config = load_config()
    
    # Kiểm tra GPU
    try:
        device = get_device()
        logging.info(f'Using device: {device}')
    except Exception as e:
        logging.error(f"Device initialization failed: {e}")
        device = torch.device("cpu")
        logging.info("Falling back to CPU")

    
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
        data_dir=os.path.join(project_root, 'data/dataset', 'train'),
        transform=train_transform
    )
    
    val_dataset = CustomDataset(
        data_dir=os.path.join(project_root, 'data/dataset', 'validation'),
        transform=val_transform
    )
    
    # Tạo weighted sampler cho tập train
    train_sampler = get_weighted_sampler(train_dataset.labels, beta=0.999)
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers = 2 if device.type == 'mps' else 4 , # Mac thường gặp lỗi với num_workers > 0,
        persistent_workers=True,
        pin_memory=(device.type == 'cuda'),  # Chỉ bật cho CUDA
        prefetch_factor=2  # NEW
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers = 2 if device.type == 'mps' else 4 , # Mac thường gặp lỗi với num_workers > 0,
        pin_memory=(device.type == 'cuda')  # Chỉ bật cho CUDA
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
        lr=1e-5,
        weight_decay=config['training']['weight_decay']
    )
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Số epoch mỗi chu kỳ
        T_mult=1,
        eta_min=1e-6  # LR tối thiểu
    )
    
    # Huấn luyện model
    model, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=config['training']['epochs'], device=device
    )
    
    logging.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    main()