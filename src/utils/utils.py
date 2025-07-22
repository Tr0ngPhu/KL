import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from typing import Dict, Any, Optional
import cv2

def ensure_dir(directory: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_acc: float, filepath: str) -> Dict[str, Any]:
    """Save model checkpoint with enhanced metadata."""
    ensure_dir(os.path.dirname(filepath))
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc,
        'model_config': {
            'image_size': getattr(model, 'image_size', 224),
            'num_classes': getattr(model, 'num_classes', 2),
            'model_type': type(model).__name__
        }
    }
    torch.save(checkpoint, filepath)
    return checkpoint

def load_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Load model checkpoint from a given path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def visualize_attention_heatmap(
    attention_weights: np.ndarray, 
    original_image: np.ndarray, 
    save_path: Optional[str] = None, 
    title: str = "Attention Heatmap"
) -> plt.Figure:
    """Visualize attention heatmap with high-quality rendering."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original Image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 2. Attention Heatmap
    # Resize attention map to match image size using a high-quality interpolation
    attention_resized = cv2.resize(attention_weights, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Normalize for better color mapping
    attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
    
    im = axes[1].imshow(attention_normalized, cmap='viridis', alpha=0.9)
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Overlayed Image
    # Convert original image to float if it's in uint8 format
    overlay_img = original_image.astype(float) / 255.0 if original_image.max() > 1 else original_image.copy()
    
    heatmap_colored = plt.cm.viridis(attention_normalized)[:, :, :3]
    overlay_combined = 0.5 * overlay_img + 0.5 * heatmap_colored
    
    axes[2].imshow(np.clip(overlay_combined, 0, 1))
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

def create_confusion_matrix_plot(y_true, y_pred, class_names, save_path=None):
    """Create and save a high-quality confusion matrix plot."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14}, cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=18, weight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, -0.05, f'Overall Accuracy: {accuracy:.4f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def analyze_model_performance(y_true, y_pred, y_prob=None, class_names=['Fake', 'Real']):
    """Provide a comprehensive analysis of model performance."""
    from sklearn.metrics import classification_report, roc_auc_score
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
        except ValueError:
            metrics['auc_roc'] = None # Handle cases with only one class in batch
    
    return metrics, report

def get_model_summary(model: torch.nn.Module) -> Dict[str, Any]:
    """Get a detailed summary of the model including parameters and size."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = {
        'total_parameters': f"{total_params:,}",
        'trainable_parameters': f"{trainable_params:,}",
        'model_size_mb': f"{total_params * 4 / (1024 * 1024):.2f} MB",
        'model_type': type(model).__name__
    }
    
    return summary