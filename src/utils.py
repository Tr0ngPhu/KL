import os
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

def ensure_dir(directory):
    """Tạo thư mục nếu chưa tồn tại"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_logging(log_file):
    """Thiết lập logging"""
    ensure_dir(os.path.dirname(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def check_admin_access():
    """Kiểm tra quyền admin"""
    try:
        return os.getuid() == 0
    except AttributeError:
        return False

def save_checkpoint(model, optimizer, epoch, path):
    """Lưu checkpoint của model"""
    ensure_dir(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, path):
    """Load checkpoint của model"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def visualize_attention(image, attention):
    """Visualize attention map và superimpose lên ảnh gốc"""
    # Giả sử attention là ma trận (197, 197) từ Vision Transformer
    # Lấy attention từ class token (token đầu tiên) đến các patch
    cls_attention = attention[0, 1:]  # Shape: (196,)
    
    # Tính grid_size từ số patch
    grid_size = int(np.sqrt(cls_attention.shape[0]))  # grid_size = 14 cho 196 patch
    if grid_size * grid_size != cls_attention.shape[0]:
        raise ValueError(f"Số patch không thể reshape thành grid vuông: {cls_attention.shape[0]}")
    
    # Reshape thành grid
    attn_map = cls_attention.reshape(grid_size, grid_size)  # Shape: (14, 14)
    
    # Resize về kích thước ảnh
    attn_map = cv2.resize(attn_map, (image.shape[1], image.shape[0]))
    
    # Chuẩn hóa giá trị về [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-5)
    
    # Áp dụng colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Chuẩn bị overlay từ ảnh gốc
    overlay = image.copy()
    # Chuẩn hóa ảnh đầu vào về [0, 1] nếu cần
    if overlay.max() > 1:
        overlay = overlay / 255.0
    # Đảm bảo overlay là np.uint8 sau khi chuyển về [0, 255]
    overlay = (overlay * 255).astype(np.uint8)
    
    # Superimpose heatmap lên ảnh gốc
    alpha = 0.5
    vis = cv2.addWeighted(heatmap, alpha, overlay, 1 - alpha, 0, dtype=cv2.CV_8U)
    
    return vis

def plot_attention_maps(images, attentions, save_path):
    """Vẽ attention maps cho một số ảnh"""
    num_images = min(len(images), 4)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
    
    for i in range(num_images):
        img = images[i].cpu().detach().numpy()  # Shape: (3, 224, 224)
        img = np.transpose(img, (1, 2, 0))  # Shape: (224, 224, 3)
        # Chuẩn hóa giá trị ảnh về [0, 1] nếu cần
        if img.max() > 1:
            img = img / 255.0
        attn = attentions[i].cpu().detach().numpy()  # Shape: (197, 197)
        
        vis_img = visualize_attention(img, attn)
        
        # Đảm bảo img trong khoảng [0, 1] cho matplotlib
        axes[i, 0].imshow(img, vmin=0, vmax=1)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(vis_img)
        axes[i, 1].set_title('Attention Map')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()