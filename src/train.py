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
    train_transform = [
        transforms.Resize((model_cfg['image_size'], model_cfg['image_size'])),
        transforms.RandomHorizontalFlip(p=aug_cfg.get('horizontal_flip', 0.5)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if aug_cfg.get('random_erasing', 0) > 0:
        train_transform.append(transforms.RandomErasing(p=aug_cfg['random_erasing']))
    val_transform = [
        transforms.Resize((model_cfg['image_size'], model_cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(train_transform), transforms.Compose(val_transform)

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
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(device_type=device.type):
                outputs = model(inputs)
            
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    report = classification_report(all_labels, all_preds, target_names=['Fake', 'Real'], output_dict=True, zero_division=0)
    return report['accuracy'], report, all_preds


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
    train_dataset = CustomDataset(os.path.join(project_root, 'src', 'data', 'train'), transform=train_transform)
    val_dataset = CustomDataset(os.path.join(project_root, 'src', 'data', 'validation'), transform=val_transform)
    
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
    criterion = FocalLoss(
        alpha=0.25,  # Optionally add to config if needed
        gamma=2.0,   # Optionally add to config if needed
        label_smoothing=config['training']['label_smoothing']
    )
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=1e-6)
    scaler = GradScaler()

    # Training Loop with Early Stopping

    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stopping_patience = config['training']['early_stopping_patience']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(project_root, 'results', timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize lists to track loss and accuracy
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"\n{'='*15} STARTING TRAINING {'='*15}")
    for epoch in range(config['training']['epochs']):
        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        train_losses.append(train_loss)

        # Validation
        val_acc, val_report, _ = validate(model, val_loader, device)
        val_accuracies.append(val_acc * 100)
        # For val_loss, use average cross-entropy on val set
        val_loss = val_report['weighted avg']['precision']  # Placeholder, will fix below
        # Compute val_loss as mean cross-entropy
        # We'll re-run validation to get logits and labels for loss
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / max(1, val_batches)
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
    _, last_report, y_pred = validate(model, val_loader, device)

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