# KL/train.py - Improved version

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
from utils import ensure_dir, check_admin_access, setup_logging, plot_attention_maps
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

class FocalLoss(nn.Module):
    """Focal Loss để xử lý class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy"""
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, preds, targets):
        n_classes = preds.size(-1)
        log_preds = nn.LogSoftmax(dim=-1)(preds)
        targets = torch.zeros_like(preds).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / n_classes
        return (-targets * log_preds).mean(0).sum()

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

def save_checkpoint(checkpoint_dict, filepath):
    """Save checkpoint - fixed version"""
    ensure_dir(os.path.dirname(filepath))
    torch.save(checkpoint_dict, filepath)

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
    def __init__(self, patience=10, min_delta=0.001, mode='max', verbose=True):
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
                if self.verbose and self.counter % 3 == 0:  # Giảm spam logs
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
                if self.verbose and self.counter % 3 == 0:  # Giảm spam logs
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
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy', color='blue')
    plt.plot(val_accs, label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot overfitting indicator
    plt.subplot(1, 3, 3)
    gap = np.array(train_accs) - np.array(val_accs)
    plt.plot(gap, label='Train-Val Gap', color='orange')
    plt.title('Overfitting Indicator')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap (%)')
    plt.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Warning Line')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def get_weighted_sampler(labels, beta=0.999):
    """Tạo weighted sampler với class balancing mạnh hơn"""
    class_counts = np.bincount(labels)
    
    # Effective number of samples
    effective_num = 1.0 - np.power(beta, class_counts)
    class_weights = (1.0 - beta) / (effective_num + 1e-6)
    
    # Chuẩn hóa weights
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    # In thông tin balancing
    logging.info(f"Class counts: {class_counts}")
    logging.info(f"Class weights: {class_weights}")
    
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device='cuda',resume_checkpoint=None):
    """Hàm huấn luyện model được cải thiện"""
    
    # Tạo thư mục lưu kết quả
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Thư mục lưu attention maps
    attention_dir = os.path.join(results_dir, 'attention_maps')
    os.makedirs(attention_dir, exist_ok=True)
    
    # Setup interrupt handlers
    setup_interrupt_handlers(model, results_dir)
    
    # Khởi tạo các biến theo dõi
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping', {}).get('patience', 10),
        mode='max'
    )
    
    # Tải checkpoint nếu có
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logging.info(f"Resumed from checkpoint at epoch {start_epoch} with best val acc: {best_val_acc:.2f}%")

    num_epochs = config['training']['epochs']
    
    # Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        if epoch % 10 == 0:  # Log system stats mỗi 10 epoch
            log_system_stats()
        
        # --- PHASE HUẤN LUYỆN ---
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            # Gradient clipping để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Cập nhật metrics
            epoch_train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Cập nhật progress bar ít thường xuyên hơn
            if batch_idx % 100 == 0:
                train_bar.set_postfix({
                    'loss': f'{epoch_train_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
        
        # Tính toán metrics huấn luyện
        train_loss = epoch_train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # --- PHASE VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Cập nhật metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Tính toán metrics validation
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Cập nhật learning rate
        if hasattr(scheduler, 'step'):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Lưu metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Log epoch results
        if epoch % 5 == 0 or epoch == num_epochs - 1:  # Log mỗi 5 epoch
            logging.info(f'Epoch {epoch+1}/{num_epochs}: '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                        f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Cảnh báo overfitting
        if train_acc - val_acc > 10:  # Gap > 10%
            logging.warning(f'Potential overfitting detected! Train-Val gap: {train_acc - val_acc:.2f}%')
        
        # Lưu model tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'best_val_acc': best_val_acc,
                'config': config
            }, os.path.join(results_dir, 'best_model.pth'))
            logging.info(f'Model tốt nhất được lưu với accuracy: {best_val_acc:.2f}%')
        
        # Lưu checkpoint định kỳ
        if epoch % 20 == 0 and epoch > 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'val_acc': val_acc,
                'config': config
            }, os.path.join(results_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # Early stopping
        early_stopping(val_acc, epoch)
        if early_stopping.early_stop:
            logging.info('Early stopping triggered')
            break
    
    # Vẽ đồ thị và lưu kết quả cuối cùng
    plot_training_history(train_losses, val_losses, train_accs, val_accs, results_dir)
    plot_confusion_matrix(all_labels, all_preds, ['Real', 'Fake'], results_dir)
    
    # Báo cáo phân loại
    report = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'])
    logging.info('\nBáo cáo phân loại:\n' + report)
    
    # Lưu báo cáo chi tiết
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    return model, best_val_acc

def load_config():
    config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_device():
    if torch.backends.mps.is_available():
        try:
            torch.mps.set_per_process_memory_fraction(0.7)  # Giảm xuống 70%
            torch.mps.empty_cache()
            return torch.device("mps")
        except Exception as e:
            logging.warning(f"Không thể thiết lập bộ nhớ MPS: {str(e)}")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main(resume_checkpoint=None):
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
    
    # Tạo transforms với augmentation mạnh hơn
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
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
        num_workers=2 if device.type == 'mps' else 4,
        persistent_workers=True,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2 if device.type == 'mps' else 4,
        pin_memory=(device.type == 'cuda')
    )
    
    # Khởi tạo model với dropout cao hơn
    model = VisionTransformer(
        image_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        num_classes=config['model']['num_classes'],
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        mlp_dim=config['model']['mlp_dim'],
        dropout=config['training'].get('dropout', 0.2)  # Tăng dropout
    ).to(device)
    
    # Loss function
    if config['training'].get('label_smoothing', 0) > 0:
        criterion = LabelSmoothingCrossEntropy(epsilon=config['training']['label_smoothing'])
    elif config.get('class_balancing', {}).get('focal_loss', {}).get('use', False):
        criterion = FocalLoss(
            alpha=config['class_balancing']['focal_loss']['alpha'],
            gamma=config['class_balancing']['focal_loss']['gamma']
        )
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer với learning rate thấp hơn
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        eps=1e-8
    )
    
    # Scheduler
    scheduler_config = config.get('scheduler', {})
    if scheduler_config.get('type') == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'max'),
            factor=float(scheduler_config.get('factor', 0.5)),
            patience=float(scheduler_config.get('patience', 5)),
            min_lr=float(scheduler_config.get('min_lr', 1e-7)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,  # Tăng chu kỳ
            T_mult=1,
            eta_min=1e-7
        )
    
    # Huấn luyện model
    model, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        config, device=device,resume_checkpoint=resume_checkpoint
    )
    
    logging.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    args = parser.parse_args()
    main(resume_checkpoint=args.resume)