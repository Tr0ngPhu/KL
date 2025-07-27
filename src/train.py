
# --- IMPORTS (đặt ở đầu file theo chuẩn Python) ---
import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import warnings
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchmetrics

# --- Wrapper cho albumentations để dùng trong transforms.Compose, tránh lỗi pickle ---
class AlbumentationsTransform:
    def __init__(self, albumentations_aug):
        self.albumentations_aug = albumentations_aug
    def __call__(self, img):
        import numpy as np
        return self.albumentations_aug(image=np.array(img))['image']

# --- Custom Augmentations (must be top-level for multiprocessing) ---
class GridMask(object):
    def __init__(self, d_min=32, d_max=96, ratio=0.6, rotate=1, mode=0, prob=0.5):
        self.d_min = d_min
        self.d_max = d_max
        self.ratio = ratio
        self.rotate = rotate
        self.mode = mode
        self.prob = prob
    def __call__(self, img):
        import torch
        # Only apply to tensor, skip if not
        if not isinstance(img, torch.Tensor):
            return img
        if random.random() > self.prob:
            return img
        h, w = img.shape[1:]
        d = random.randint(self.d_min, self.d_max)
        l = int(d * self.ratio + 0.5)
        st_h = random.randint(0, d)
        st_w = random.randint(0, d)
        mask = torch.ones((h, w), dtype=img.dtype, device=img.device)
        for i in range(-1, h//d+1):
            s = d * i + st_h
            t = s + l
            s = max(s, 0)
            t = min(t, h)
            mask[s:t, :] = 0
        for i in range(-1, w//d+1):
            s = d * i + st_w
            t = s + l
            s = max(s, 0)
            t = min(t, w)
            mask[:, s:t] = 0
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(img)
        return img * mask

class RandomGaussianBlur(object):
    def __init__(self, p=0.3, kernel_size=3):
        self.p = p
        self.kernel_size = kernel_size
    def __call__(self, img):
        import torchvision.transforms.functional as F
        if random.random() < self.p:
            return F.gaussian_blur(img, kernel_size=self.kernel_size)
        return img

class RandomNoise(object):
    def __init__(self, p=0.3, std=0.05):
        self.p = p
        self.std = std
    def __call__(self, img):
        if random.random() < self.p:
            noise = torch.randn_like(img) * self.std
            return img + noise
        return img
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from models.model import VisionTransformer
import yaml
import warnings
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchmetrics

# --- Pre-computation and Path Setup ---
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.data.dataset import CustomDataset
from src.utils.utils import save_checkpoint, create_confusion_matrix_plot

# --- Optimized Loss Function ---
class FocalLoss(nn.Module):
    """Focal Loss with Label Smoothing for better generalization and handling of hard examples."""
    def __init__(self, alpha, gamma, label_smoothing, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            targets = (1.0 - self.label_smoothing) * F.one_hot(targets, num_classes=num_classes) + self.label_smoothing / num_classes
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# --- Data Augmentation ---
def get_optimized_transforms():
    """Advanced and efficient data augmentation techniques."""
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    aug_cfg = config['augmentation']
    model_cfg = config['model']
    # Albumentations augmentations (không có RandomErasing)
    albumentations_aug = A.Compose([
        A.Resize(model_cfg['image_size'], model_cfg['image_size']),
        A.HorizontalFlip(p=aug_cfg.get('horizontal_flip', 0.5)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    train_transform = transforms.Compose([
        AlbumentationsTransform(albumentations_aug),
        transforms.RandomErasing(p=aug_cfg.get('random_erasing', 0.1)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((model_cfg['image_size'], model_cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

def get_weighted_sampler(labels):
    """Effective sampler to handle class imbalance."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# --- Training and Validation Loops ---
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    total = len(loader)
    for idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # --- DEBUG: Print batch statistics for first batch ---
        if idx == 0:
            print("\n[DEBUG] Batch 0 stats:")
            print(f"  inputs.shape: {inputs.shape}")
            print(f"  inputs.min: {inputs.min().item():.4f}, max: {inputs.max().item():.4f}, mean: {inputs.mean().item():.4f}, std: {inputs.std().item():.4f}")
            print(f"  labels: {labels.tolist()}")
            from collections import Counter
            print(f"  label distribution: {Counter(labels.tolist())}")

        optimizer.zero_grad()
        with autocast(device_type=device.type):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        percent = int((idx + 1) / total * 100)
        print(f"Training: {percent}%/{100}% - Loss: {loss.item():.4f}      ", end='\r')
    print()  # Newline after epoch
    return total_loss / total

def validate(model, loader, device, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    val_loss_sum = 0.0
    val_batches = 0
    total = len(loader)
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # --- DEBUG: Print batch statistics for first batch ---
            if idx == 0:
                print("\n[DEBUG] Val Batch 0 stats:")
                print(f"  inputs.shape: {inputs.shape}")
                print(f"  inputs.min: {inputs.min().item():.4f}, max: {inputs.max().item():.4f}, mean: {inputs.mean().item():.4f}, std: {inputs.std().item():.4f}")
                print(f"  labels: {labels.tolist()}")
                from collections import Counter
                print(f"  label distribution: {Counter(labels.tolist())}")

            with autocast(device_type=device.type):
                outputs = model(inputs)
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    val_loss_sum += loss.item()
                    val_batches += 1
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            percent = int((idx + 1) / total * 100)
            print(f"Validating: {percent}%/{100}%      ", end='\r')
    print()  # Newline after val
    # Tính metric bằng torchmetrics
    all_labels_tensor = torch.tensor(all_labels).to(device)
    all_preds_tensor = torch.tensor(all_preds).to(device)
    metric_acc = torchmetrics.classification.BinaryAccuracy().to(device)
    metric_f1 = torchmetrics.classification.BinaryF1Score().to(device)
    acc = metric_acc(all_preds_tensor, all_labels_tensor).item()
    f1 = metric_f1(all_preds_tensor, all_labels_tensor).item()
    report = classification_report(all_labels, all_preds, target_names=['Fake', 'Real'], output_dict=True, zero_division=0)
    val_loss = val_loss_sum / max(1, val_batches) if criterion is not None else None
    return acc, f1, report, all_preds, val_loss


# --- Main Function ---
def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("❌ GPU is required for this optimized training script.")
        return
    
    print(f"Training on: {torch.cuda.get_device_name(0)}")
    
    # Config
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Data
    train_transform, val_transform = get_optimized_transforms()
    train_dataset = CustomDataset(r'D:\KL-1\src\data\train', transform=train_transform, use_albumentations=True)
    val_dataset = CustomDataset(os.path.join(project_root, 'data', 'validation'), transform=val_transform, use_albumentations=False)

    from collections import Counter
    train_counter = Counter(train_dataset.labels)
    val_counter = Counter(val_dataset.labels)
    print("[DEBUG] Train class distribution:", dict(train_counter))
    print("[DEBUG] Val class distribution:", dict(val_counter))

    train_sampler = get_weighted_sampler(train_dataset.labels)

    # Balance batch size and performance for typical gaming GPUs (e.g., RTX 3060/4060)
    batch_size = config['training']['batch_size']

    num_workers = config['training']['num_workers']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    print(f"📊 Datasets: {len(train_dataset)} train, {len(val_dataset)} validation images.")
    print(f"⚡ Batch size: {batch_size}, Num workers: 4")

    # Model ViT custom
    model = VisionTransformer(
        img_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        in_channels=3,
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        dropout=config['model']['dropout'],
        drop_path_rate=config['model']['drop_path_rate'],
        use_cls_token=True,
        use_se=config['model'].get('use_se', False)
    ).to(device)
    print(f"🧠 Model: VisionTransformer ({sum(p.numel() for p in model.parameters()):,} params)")

    # Training Components
    # Tính class weights cho CrossEntropyLoss
    train_labels = train_dataset.labels
    from collections import Counter
    label_counts = Counter(train_labels)
    num_classes = config['model']['num_classes']
    class_sample_count = [label_counts.get(i, 0) for i in range(num_classes)]
    class_weights = [0.0 if c == 0 else sum(class_sample_count) / (num_classes * c) for c in class_sample_count]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=config['training']['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=1e-6)
    scaler = GradScaler()

    # Training Loop with Early Stopping

    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stopping_patience = config['training']['early_stopping_patience']
    timestamp = None
    results_dir = None

    # Initialize lists to track loss and accuracy
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"\n{'='*15} STARTING TRAINING {'='*15}")
    # Tạo thư mục results chỉ khi bắt đầu train thành công (epoch 0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(project_root, 'results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    for epoch in range(config['training']['epochs']):
        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        train_losses.append(train_loss)

        # Validation (single call, returns acc, report, preds, val_loss)
        val_acc, val_f1, val_report, _, val_loss = validate(model, val_loader, device, criterion)
        val_accuracies.append(val_acc * 100)
        val_losses.append(val_loss)

        # For train accuracy, run on a batch (approximate, for speed)
        model.eval()
        train_acc_sum = 0.0
        train_batches = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                train_acc_sum += (preds == labels).float().mean().item()
                train_batches += 1
                if train_batches >= 5:
                    break  # Only sample a few batches for speed
        train_acc = train_acc_sum / max(1, train_batches)
        train_accuracies.append(train_acc * 100)

        scheduler.step()

        print(f"Epoch {epoch+1:2d}/{config['training']['epochs']} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.2%} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}", end="")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_acc, os.path.join(results_dir, 'best_model.pth'))
            print(" ✨ New best model saved!")
        else:
            epochs_no_improve += 1
            print()

        # Early stopping check
        if epochs_no_improve >= early_stopping_patience:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs. No improvement for {early_stopping_patience} epochs.")
            break

    # Final evaluation
    print(f"\n{'='*15} TRAINING COMPLETE {'='*15}")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2%}")

    # Load the best model for final evaluation
    best_model_path = os.path.join(results_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📊 Loaded best model from epoch {checkpoint['epoch']+1} for final evaluation.")

    # Get final predictions and report
    y_true = val_dataset.labels
    _, _, last_report, y_pred, _ = validate(model, val_loader, device, criterion)

    # Create and save confusion matrix
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    create_confusion_matrix_plot(y_true, y_pred, ['Fake', 'Real'], save_path=cm_path)
    print(f"📈 Confusion matrix saved to {cm_path}")

    # Save training curve (loss/acc) to results dir
    try:
        import matplotlib.pyplot as plt
        epochs = range(1, len(train_losses)+1)
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        # Loss
        axs[0].plot(epochs, train_losses, 'b-', label='Train Loss')
        axs[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
        axs[0].set_title('Training and Validation Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        # Accuracy
        axs[1].plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
        axs[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        axs[1].set_title('Training and Validation Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].legend()
        # Overfitting indicator
        acc_gap = [v-t for t,v in zip(train_accuracies, val_accuracies)]
        axs[2].plot(epochs, acc_gap, color='orange', label='Train-Val Gap')
        axs[2].axhline(5, color='red', linestyle='--', label='Warning Line')
        axs[2].set_title('Overfitting Indicator')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Accuracy Gap (%)')
        axs[2].legend()
        plt.tight_layout()
        curve_path = os.path.join(results_dir, 'training_curve.png')
        plt.savefig(curve_path, dpi=200)
        plt.close()
        print(f"📊 Training curve saved to {curve_path}")
    except Exception as e:
        print(f"⚠️ Could not save training curve: {e}")

    print("\nFinal Classification Report:")
    print(f"  Precision: {last_report['weighted avg']['precision']:.2f}")
    print(f"  Recall: {last_report['weighted avg']['recall']:.2f}")
    print(f"  F1-Score: {last_report['weighted avg']['f1-score']:.2f}")
    print(f"\nResults saved in: {results_dir}")

    # Ghi log kết quả training vào training.log (chỉ ghi khi kết thúc toàn bộ train hoặc early stop)
    log_path = os.path.join(project_root, config['paths']['log_file'])
    with open(log_path, 'a', encoding='utf-8') as log_file:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        log_file.write(f"{now} - INFO - Training completed\n")
        log_file.write(f"{now} - INFO - Best Validation Accuracy: {best_val_acc:.2%}\n")
        log_file.write(f"{now} - INFO - Model: VisionTransformer\n")
        log_file.write(f"{now} - INFO - Training parameters: batch_size={batch_size}, learning_rate={config['training']['learning_rate']}, epochs={config['training']['epochs']}, dropout={config['model']['dropout']}, attn_drop_rate={config['model']['drop_path_rate']}, mixup={config['training'].get('use_mixup', False)}, cutmix={config['training'].get('use_cutmix', False)}, label_smoothing={config['training'].get('label_smoothing', 0.0)}\n")
        log_file.write(f"{now} - INFO - Dataset: {len(train_dataset)} train, {len(val_dataset)} val (Fake: {train_dataset.labels.count(0)}, Real: {train_dataset.labels.count(1)})\n")

if __name__ == '__main__':
    main()