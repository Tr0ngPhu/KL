import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

def generate_attention_heatmap(
    image, attn_map, save_path=None, colormap='turbo', alpha=0.45, blur_ksize=0,
    norm_type='minmax', return_array=False
):
    """
    Sinh heatmap đẹp, làm mượt, overlay lên ảnh gốc hoặc trả về heatmap array.
    colormap: tên colormap matplotlib ('turbo', 'plasma', 'jet', ...)
    alpha: độ trong suốt của heatmap
    blur_ksize: kernel size cho Gaussian blur (0: không làm mượt)
    norm_type: 'minmax' hoặc 'zscore'
    return_array: True -> chỉ trả về heatmap array (không vẽ/lưu ảnh)
    """
    import os
    if isinstance(image, Image.Image):
        image = np.array(image)
    # Chuẩn hóa heatmap
    if norm_type == 'zscore':
        mean = np.nanmean(attn_map)
        std = np.nanstd(attn_map) + 1e-8
        heatmap = (attn_map - mean) / std
        heatmap = (heatmap - np.nanmin(heatmap)) / (np.nanmax(heatmap) - np.nanmin(heatmap) + 1e-8)
    else:
        heatmap = (attn_map - np.nanmin(attn_map)) / (np.nanmax(attn_map) - np.nanmin(attn_map) + 1e-8)
    heatmap = np.nan_to_num(heatmap)
    # Resize heatmap về đúng kích thước ảnh
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    # Làm mượt heatmap nếu blur_ksize > 0
    if blur_ksize and blur_ksize > 1:
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (blur_ksize|1, blur_ksize|1), 0)
    if return_array:
        return heatmap_resized
    # Overlay heatmap màu nhiệt lên ảnh gốc
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(heatmap_resized, cmap=colormap, alpha=alpha)
    plt.axis('off')
    plt.title("AI Focus Heatmap", fontsize=16, weight='bold')
    if save_path is None:
        uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(uploads_dir, f'heatmap_{timestamp}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
    # Đảm bảo file đã được ghi ra
    import time
    for _ in range(10):
        if os.path.exists(save_path):
            break
        time.sleep(0.05)
    return save_path

def get_focus_region(heatmap, image, patch_size=16, topk=1, return_mask=False):
    """
    Lấy vùng tập trung nhất hoặc top-k vùng từ heatmap trên ảnh.
    Trả về: patch ảnh, vị trí (y, x) hoặc mask các vùng focus.
    """
    h, w = image.shape[:2]
    flat = heatmap.flatten()
    idxs = np.argpartition(flat, -topk)[-topk:]
    coords = [np.unravel_index(idx, heatmap.shape) for idx in idxs]
    if return_mask:
        mask = np.zeros_like(heatmap, dtype=np.uint8)
        for y, x in coords:
            y1 = max(0, y - patch_size)
            y2 = min(h, y + patch_size)
            x1 = max(0, x - patch_size)
            x2 = min(w, x + patch_size)
            mask[y1:y2, x1:x2] = 1
        return mask, coords
    # Trả về patch trung bình các vùng focus
    patches = []
    for y, x in coords:
        y1 = max(0, y - patch_size)
        y2 = min(h, y + patch_size)
        x1 = max(0, x - patch_size)
        x2 = min(w, x + patch_size)
        patch = image[y1:y2, x1:x2, :] if image.ndim == 3 else image[y1:y2, x1:x2]
        patches.append(patch)
    if len(patches) == 1:
        return patches[0], coords[0]
    # Trả về patch trung bình
    min_shape = np.min([p.shape[:2] for p in patches], axis=0)
    patches_resized = [p[:min_shape[0], :min_shape[1]] for p in patches]
    avg_patch = np.mean(patches_resized, axis=0)
    return avg_patch, coords

def explain_focus_region(focus_patch, predicted_class):
    """
    Sinh lời giải thích dựa trên vùng AI tập trung, bổ sung entropy, hue, texture.
    """
    brightness = np.mean(focus_patch)
    contrast = np.std(focus_patch)
    color = np.mean(focus_patch, axis=(0,1)) if focus_patch.ndim==3 else [0,0,0]
    # Entropy
    from scipy.stats import entropy
    hist = np.histogram(focus_patch, bins=32)[0]
    ent = entropy(hist+1e-8)
    # Hue (nếu có màu)
    hue = 0
    if focus_patch.ndim==3:
        hsv = cv2.cvtColor(focus_patch.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hue = np.mean(hsv[:,:,0])
    # Texture (std của gradient)
    grad = np.gradient(focus_patch.astype(float), axis=(0,1))
    grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
    texture = np.std(grad_mag)
    explanation = (
        f"AI tập trung vào vùng có độ sáng {brightness:.1f}, độ tương phản {contrast:.1f}, màu sắc RGB {color}, "
        f"entropy {ent:.2f}, sắc độ (hue) {hue:.1f}, texture {texture:.2f}. "
        f"Đây là vùng quan trọng nhất để phân biệt {predicted_class.lower()} trên ảnh này."
    )
    return explanation
