import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from typing import Dict, Any, Optional
import cv2

def ensure_dir(directory: str):
    """
    Tạo thư mục nếu chưa tồn tại (tối ưu cho lưu checkpoint, hình ảnh).
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_acc: float, filepath: str) -> Dict[str, Any]:
    """
    Lưu checkpoint mô hình với metadata chi tiết (tối ưu cho việc khôi phục).
    """
    ensure_dir(os.path.dirname(filepath))
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc,
        'model_config': {
            'image_size': getattr(model, 'image_size', None),
            'num_classes': getattr(model, 'num_classes', None),
            'model_type': type(model).__name__
        }
    }
    torch.save(checkpoint, filepath)
    return checkpoint

def load_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Khôi phục mô hình từ checkpoint (tối ưu cho việc kiểm tra và inference).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

## Đã chuyển toàn bộ hàm heatmap sang heatmap_utils.py để tập trung xử lý AI focus.

def create_confusion_matrix_plot(y_true, y_pred, class_names, save_path=None):
    """
    Vẽ và lưu confusion matrix chất lượng cao (tối ưu cho báo cáo).
    """
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
    plt.figtext(0.5, -0.05, f'Độ chính xác tổng thể: {accuracy:.4f}', ha='center', fontsize=12)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return plt.gcf()

def analyze_model_performance(y_true, y_pred, y_prob=None, class_names=['Fake', 'Real']):
    """
    Phân tích hiệu năng mô hình (tối ưu cho báo cáo và giám sát).
    """
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
            metrics['auc_roc'] = None # Trường hợp chỉ có một class
    return metrics, report

def get_model_summary(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Tóm tắt chi tiết mô hình (tham số, kích thước, loại mô hình).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {
        'total_parameters': f"{total_params:,}",
        'trainable_parameters': f"{trainable_params:,}",
        'model_size_mb': f"{total_params * 4 / (1024 * 1024):.2f} MB",
        'model_type': type(model).__name__
    }
    return summary