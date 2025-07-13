# KL/train.py - Enhanced Version for Training ExplainableVisionTransformer

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import yaml
import signal
import psutil
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Thêm đường dẫn gốc vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.explainable_model import ExplainableVisionTransformer  # Đảm bảo dùng đúng mô hình
from src.utils import ensure_dir, check_admin_access, setup_logging, plot_attention_maps
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
                if self.verbose and self.counter % 3 == 0:
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
                if self.verbose and self.counter % 3 == 0:
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
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy', color='blue')
    plt.plot(val_accs, label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
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

def get_weighted_sampler(labels, beta=0.9999):
    class_counts = np.bincount(labels)
    effective_num = 1.0 - np.power(beta, class_counts)
    class_weights = (1.0 - beta) / (effective_num + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    logging.info(f"Class counts: {class_counts}")
    logging.info(f"Class weights: {class_weights}")
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def validate_config(config):
    required_keys = [
        ('training.learning_rate', float),
        ('training.weight_decay', float),
        ('training.batch_size', int),
        ('training.epochs', int),
        ('model.image_size', int),
        ('model.patch_size', int),
        ('model.num_classes', int),
        ('model.dim', int),
        ('model.depth', int),
        ('model.heads', int),
        ('model.mlp_dim', int)
    ]
    for key, expected_type in required_keys:
        keys = key.split('.')
        value = config
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                raise KeyError(f"Missing or invalid config key: {key}")
            value = value[k]
        try:
            value = expected_type(value)
        except (ValueError, TypeError):
            raise ValueError(f"Config key {key} must be convertible to {expected_type.__name__}, got {value}")
    
    if config.get('scheduler', {}).get('type') == 'ReduceLROnPlateau':
        scheduler_keys = [('scheduler.factor', float), ('scheduler.patience', int), ('scheduler.min_lr', float)]
        for key, expected_type in scheduler_keys:
            keys = key.split('.')
            value = config
            for k in keys:
                if not isinstance(value, dict) or k not in value:
                    raise KeyError(f"Missing or invalid config key: {key}")
                value = value[k]
            try:
                value = expected_type(value)
                current = config
                for k in keys[:-1]:
                    current = current[k]
                current[keys[-1]] = value
            except (ValueError, TypeError):
                raise ValueError(f"Config key {key} must be convertible to {expected_type.__name__}, got {value}")

def fix_scheduler_state_dict(state_dict):
    if 'min_lrs' in state_dict:
        state_dict['min_lrs'] = [float(lr) for lr in state_dict['min_lrs']]
    if 'factor' in state_dict:
        state_dict['factor'] = float(state_dict['factor'])
    return state_dict

def check_dataset_balance(data_dir):
    """Kiểm tra số lượng ảnh trong các lớp Real và Fake"""
    real_dir = os.path.join(data_dir, 'Real')
    fake_dir = os.path.join(data_dir, 'Fake')
    
    real_count = len([f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]) if os.path.exists(real_dir) else 0
    fake_count = len([f for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))]) if os.path.exists(fake_dir) else 0
    
    logging.info(f"Dataset stats - Real: {real_count} images, Fake: {fake_count} images")
    if real_count > fake_count * 1.2 or fake_count > real_count * 1.2:
        logging.warning(f"Dataset imbalance detected: Real/Fake ratio = {real_count/fake_count:.2f}")
    return real_count, fake_count

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device='cuda', resume_checkpoint=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    attention_dir = os.path.join(results_dir, 'attention_maps')
    os.makedirs(attention_dir, exist_ok=True)
    
    setup_interrupt_handlers(model, results_dir)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping', {}).get('patience', 10),
        mode='max'
    )
    
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        try:
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(fix_scheduler_state_dict(checkpoint['scheduler_state_dict']))
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_acc = checkpoint.get('best_val_acc', checkpoint.get('val_acc', 0.0))
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            train_accs = checkpoint.get('train_accs', [])
            val_accs = checkpoint.get('val_accs', [])
            if checkpoint.get('early_stopping_state'):
                early_stopping.best_value = checkpoint['early_stopping_state'].get('best_value')
                early_stopping.counter = checkpoint['early_stopping_state'].get('counter', 0)
                early_stopping.best_epoch = checkpoint['early_stopping_state'].get('best_epoch', 0)
            logging.info(f"Resumed from checkpoint at epoch {start_epoch} with best val acc: {best_val_acc:.2f}%")
            logging.info(f"Scheduler min_lr: {scheduler.min_lrs}, factor: {scheduler.factor}")
        except KeyError as e:
            logging.error(f"Checkpoint missing key: {e}. Starting training from scratch.")
            start_epoch = 0
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
    
    num_epochs = config['training']['epochs']
    
    for epoch in range(start_epoch, num_epochs):
        if epoch % 5 == 0:
            log_system_stats()
        
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, attentions = model(inputs, return_attention=True)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                train_bar.set_postfix({
                    'loss': f'{epoch_train_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
        
        train_loss = epoch_train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        if epoch % config['visualization']['attention']['save_freq'] == 0:
            if isinstance(attentions, list):
                last_layer_attn = attentions[-1]  # [batch_size, num_heads, 197, 197]
                selected_attn = last_layer_attn[:, 0, :, :]  # [batch_size, 197, 197], chọn head đầu tiên
            else:
                selected_attn = attentions[:, 0, :, :]  # Điều chỉnh theo cấu trúc thực tế
            logging.info(f"Shape of selected_attn: {selected_attn.shape}")  # Thêm dòng này để debug
            plot_attention_maps(inputs[:4], selected_attn[:4], 
                                save_path=os.path.join(attention_dir, f'attention_epoch_{epoch}.png'))
        
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
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        if hasattr(scheduler, 'step'):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if epoch % config['training']['save_interval'] == 0 or epoch == num_epochs - 1:
            logging.info(f'Epoch {epoch+1}/{num_epochs}: '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                        f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if train_acc - val_acc > 10:
            logging.warning(f'Potential overfitting detected! Train-Val gap: {train_acc - val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'best_val_acc': best_val_acc,
                'val_acc': val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'early_stopping_state': {
                    'best_value': early_stopping.best_value,
                    'counter': early_stopping.counter,
                    'best_epoch': early_stopping.best_epoch
                },
                'config': config
            }, os.path.join(results_dir, 'best_model.pth'))
            logging.info(f'Model tốt nhất được lưu với accuracy: {best_val_acc:.2f}%')
        
        if epoch % config['training']['save_interval'] == 0 and epoch > 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'early_stopping_state': {
                    'best_value': early_stopping.best_value,
                    'counter': early_stopping.counter,
                    'best_epoch': early_stopping.best_epoch
                },
                'config': config
            }, os.path.join(results_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        early_stopping(val_acc, epoch)
        if early_stopping.early_stop:
            logging.info('Early stopping triggered')
            break
    
    plot_training_history(train_losses, val_losses, train_accs, val_accs, results_dir)
    plot_confusion_matrix(all_labels, all_preds, ['Real', 'Fake'], results_dir)
    
    report = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'])
    logging.info('\nBáo cáo phân loại:\n' + report)
    
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
            torch.mps.set_per_process_memory_fraction(0.9)
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
    config = load_config()
    validate_config(config)
    
    logging.info(f"Learning rate: {config['training']['learning_rate']}, Type: {type(config['training']['learning_rate'])}")
    logging.info(f"Min learning rate: {config.get('scheduler', {}).get('min_lr', 1e-7)}, Type: {type(config.get('scheduler', {}).get('min_lr', 1e-7))}")
    logging.info(f"Scheduler factor: {config.get('scheduler', {}).get('factor', 0.5)}, Type: {type(config.get('scheduler', {}).get('factor', 0.5))}")
    
    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint file {resume_checkpoint} does not exist")
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')
        logging.info(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    try:
        device = get_device()
        logging.info(f'Using device: {device}')
    except Exception as e:
        logging.error(f"Device initialization failed: {e}")
        device = torch.device("cpu")
        logging.info("Falling back to CPU")
    
    train_data_dir = os.path.join(project_root, 'data/dataset', 'train')
    val_data_dir = os.path.join(project_root, 'data/dataset', 'validation')
    if not os.path.exists(train_data_dir) or not os.listdir(train_data_dir):
        raise FileNotFoundError(f"Training data directory {train_data_dir} is missing or empty")
    if not os.path.exists(val_data_dir) or not os.listdir(val_data_dir):
        raise FileNotFoundError(f"Validation data directory {val_data_dir} is missing or empty")
    
    # Kiểm tra số lượng ảnh
    real_count, fake_count = check_dataset_balance(train_data_dir)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CustomDataset(
        data_dir=train_data_dir,
        transform=train_transform
    )
    
    val_dataset = CustomDataset(
        data_dir=val_data_dir,
        transform=val_transform
    )
    
    train_sampler = get_weighted_sampler(train_dataset.labels, beta=config['class_balancing']['beta'])
    
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
    
    model = ExplainableVisionTransformer(
        image_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        num_classes=config['model']['num_classes'],
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        mlp_dim=config['model']['mlp_dim'],
        dropout=config['training'].get('dropout', 0.2)
    ).to(device)
    
    if config['training'].get('label_smoothing', 0) > 0:
        criterion = LabelSmoothingCrossEntropy(epsilon=config['training']['label_smoothing'])
    elif config.get('class_balancing', {}).get('focal_loss', {}).get('use', False):
        criterion = FocalLoss(
            alpha=config['class_balancing']['focal_loss']['alpha'],
            gamma=config['class_balancing']['focal_loss']['gamma']
        )
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        eps=1e-8
    )
    
    scheduler_config = config.get('scheduler', {})
    if scheduler_config.get('type') == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'max'),
            factor=float(scheduler_config.get('factor', 0.2)),
            patience=int(scheduler_config.get('patience', 8)),
            min_lr=float(scheduler_config.get('min_lr', 1e-8)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,
            T_mult=1,
            eta_min=1e-8
        )
    logging.info(f"Initial scheduler min_lr: {scheduler.min_lrs}, factor: {scheduler.factor}")
    
    model, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        config, device=device, resume_checkpoint=resume_checkpoint
    )
    
    logging.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    args = parser.parse_args()
    main(resume_checkpoint=args.resume)