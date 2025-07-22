import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import timm
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
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1, reduction='mean'):
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
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(), # Efficient and powerful augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
    ]), transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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
    
    print(f"🚀 Training on: {torch.cuda.get_device_name(0)}")
    
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
    batch_size = config['training'].get('batch_size', 16)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    print(f"📊 Datasets: {len(train_dataset)} train, {len(val_dataset)} validation images.")
    print(f"⚡ Batch size: {batch_size}, Num workers: 4")

    # Model
    model_name = config['model'].get('name', 'vit_base_patch16_224')
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=config['model']['num_classes'],
        drop_rate=config['model'].get('dropout', 0.2),
        attn_drop_rate=config['model'].get('attn_dropout', 0.1)
    ).to(device)
    
    print(f"🧠 Model: {model_name} ({sum(p.numel() for p in model.parameters()):,} params)")

    # Training Components
    criterion = FocalLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=1e-6)
    scaler = GradScaler()

    # Training Loop
    best_val_acc = 0.0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(project_root, 'results', timestamp)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*15} STARTING TRAINING {'='*15}")
    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_acc, _, _ = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1:2d}/{config['training']['epochs']} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.2%} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}", end="")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, os.path.join(results_dir, 'best_model.pth'))
            print(" ✨ New best model saved!")
        else:
            print()

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

    print("\nFinal Classification Report:")
    print(f"  Precision: {last_report['weighted avg']['precision']:.2f}")
    print(f"  Recall: {last_report['weighted avg']['recall']:.2f}")
    print(f"  F1-Score: {last_report['weighted avg']['f1-score']:.2f}")
    print(f"\nResults saved in: {results_dir}")

if __name__ == '__main__':
    main()