# explainable_model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.cluster import KMeans
from torchvision import transforms
import json
from typing import Dict, List, Tuple, Optional

class ExplainableVisionTransformer(nn.Module):
    # Giữ nguyên mã hiện tại
    def __init__(self, image_size=224, patch_size=16, num_classes=2, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.blocks = nn.ModuleList([
            TransformerBlockWithAttention(
                dim=dim,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.grid_size = int(np.sqrt(num_patches))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x, return_attention=False):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        all_attentions = []
        for i, blk in enumerate(self.blocks):
            if return_attention:
                x, attention = blk(x, return_attention=True)
                all_attentions.append(attention)
            else:
                x = blk(x, return_attention=False)
        x = self.norm(x)
        output = self.head(x[:, 0])
        if return_attention:
            return output, all_attentions
        return output

class TransformerBlockWithAttention(nn.Module):
    # Giữ nguyên mã hiện tại
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithWeights(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attention_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.norm1(x), return_attention=False)
        x = x + self.mlp(self.norm2(x))
        if return_attention:
            return x, attention_weights
        return x

class MultiHeadAttentionWithWeights(nn.Module):
    # Giữ nguyên mã hiện tại
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5
        
    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        Q = self.w_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        output = self.w_o(context)
        if return_attention:
            return output, attention_weights
        return output

class ExplainabilityAnalyzer:
    def __init__(self, model: ExplainableVisionTransformer, class_names: List[str] = None):
        self.model = model
        self.class_names = class_names or ['Real', 'Fake']
        self.device = next(model.parameters()).device
        self.region_definitions = {
            'logo': {'position': 'top_center', 'importance': 0.9},
            'color': {'position': 'overall', 'importance': 0.7},
            'texture': {'position': 'center', 'importance': 0.8},
            'edges': {'position': 'boundary', 'importance': 0.6},
            'pattern': {'position': 'overall', 'importance': 0.7},
            'material': {'position': 'surface', 'importance': 0.8}
        }
    
    def predict_with_explanation(self, image: torch.Tensor, original_image: np.ndarray = None) -> Dict:
        self.model.eval()
        with torch.no_grad():
            logits, all_attentions = self.model(image.unsqueeze(0), return_attention=True)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            attention_analysis = self._analyze_attention_patterns(all_attentions)
            # Sửa image_shape để dùng kích thước ảnh gốc
            heatmap = self._generate_attention_heatmap(all_attentions, original_image.shape[:2])  # [height, width]
            region_analysis = self._analyze_important_regions(heatmap, attention_analysis)
            explanation = self._generate_text_explanation(predicted_class, confidence, region_analysis, attention_analysis)
            return {
                'prediction': self.class_names[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy(),
                'explanation': explanation,
                'attention_heatmap': heatmap,
                'region_analysis': region_analysis,
                'attention_stats': attention_analysis
            }
    
    def _analyze_attention_patterns(self, all_attentions: List[torch.Tensor]) -> Dict:
        combined_attention = []
        for attn in all_attentions:
            cls_attention = attn[0, :, 0, 1:].cpu().numpy()  # [heads, num_patches]
            mean_attention = np.mean(cls_attention, axis=0)
            mean_attention = (mean_attention - mean_attention.min()) / (mean_attention.max() - mean_attention.min() + 1e-8)
            combined_attention.append(mean_attention)
        
        mean_attention = np.mean(combined_attention, axis=0)
        max_attention = np.max(combined_attention, axis=0)
        std_attention = np.std(combined_attention, axis=0)
        top_patches_idx = np.argsort(mean_attention)[-5:][::-1]
        attention_entropy = -np.sum(mean_attention * np.log(mean_attention + 1e-8))
        return {
            'mean_attention': mean_attention,
            'max_attention': max_attention,
            'std_attention': std_attention,
            'top_patches': top_patches_idx.tolist(),
            'attention_entropy': attention_entropy,
            'focus_concentration': np.max(mean_attention) / (np.mean(mean_attention) + 1e-8)
        }
    
    def _generate_attention_heatmap(self, all_attentions: List[torch.Tensor], image_shape: Tuple) -> np.ndarray:
        import logging
        logger = logging.getLogger(__name__)
        try:
            if not all_attentions or not isinstance(all_attentions, list):
                raise ValueError("all_attentions phải là danh sách các tensor attention")
            
            combined_attention = []
            for attn in all_attentions:
                cls_attention = attn[0, :, 0, 1:].cpu().numpy()  # [heads, num_patches]
                mean_attention = np.mean(cls_attention, axis=0)  # [num_patches]
                mean_attention = (mean_attention - mean_attention.min()) / (mean_attention.max() - mean_attention.min() + 1e-8)
                combined_attention.append(mean_attention)
            
            mean_attention = np.mean(combined_attention, axis=0)
            logger.info(f"mean_attention shape: {mean_attention.shape}")
            
            # Kiểm tra num_patches
            num_patches = mean_attention.shape[0]
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size != num_patches:
                logger.warning(f"num_patches {num_patches} không phải số chính phương, điều chỉnh grid_size")
                grid_size = int(np.ceil(np.sqrt(num_patches)))
                # Đệm mean_attention nếu cần
                pad_size = grid_size * grid_size - num_patches
                if pad_size > 0:
                    mean_attention = np.pad(mean_attention, (0, pad_size), mode='constant', constant_values=0)
            
            # Reshape thành 2D
            attention_map = mean_attention.reshape(grid_size, grid_size)
            logger.info(f"attention_map shape: {attention_map.shape}")
            
            # Resize về kích thước ảnh gốc
            heatmap = cv2.resize(attention_map, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
            logger.info(f"heatmap shape: {heatmap.shape}")
            
            # Chuẩn hóa
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap
        except Exception as e:
            logger.error(f"Lỗi trong _generate_attention_heatmap: {str(e)}")
            raise
    
    def _analyze_important_regions(self, heatmap: np.ndarray, attention_stats: Dict) -> Dict:
        h, w = heatmap.shape
        regions = {
            'top_left': heatmap[:h//2, :w//2],
            'top_center': heatmap[:h//2, w//4:3*w//4],
            'top_right': heatmap[:h//2, w//2:],
            'center_left': heatmap[h//4:3*h//4, :w//2],
            'center': heatmap[h//4:3*h//4, w//4:3*w//4],
            'center_right': heatmap[h//4:3*h//4, w//2:],
            'bottom_left': heatmap[h//2:, :w//2],
            'bottom_center': heatmap[h//2:, w//4:3*w//4],
            'bottom_right': heatmap[h//2:, w//2:],
            'edges': np.concatenate([
                heatmap[0, :], heatmap[-1, :], 
                heatmap[:, 0], heatmap[:, -1]
            ])
        }
        region_scores = {}
        for region_name, region_data in regions.items():
            if region_data.size > 0:
                region_scores[region_name] = {
                    'mean_intensity': np.mean(region_data),
                    'max_intensity': np.max(region_data),
                    'coverage': np.sum(region_data > 0.5) / region_data.size
                }
            else:
                region_scores[region_name] = {
                    'mean_intensity': 0.0,
                    'max_intensity': 0.0,
                    'coverage': 0.0
                }
        top_regions = sorted(region_scores.items(), 
                            key=lambda x: x[1]['mean_intensity'], reverse=True)[:3]
        return {
            'region_scores': region_scores,
            'top_regions': top_regions,
            'focus_distribution': self._classify_attention_pattern(heatmap)
        }
    
    def _classify_attention_pattern(self, heatmap: np.ndarray) -> str:
        h, w = heatmap.shape
        center_region = heatmap[h//4:3*h//4, w//4:3*w//4]
        edge_pixels = np.concatenate([
            heatmap[0, :], heatmap[-1, :], 
            heatmap[:, 0], heatmap[:, -1]
        ])
        center_intensity = np.mean(center_region)
        edge_intensity = np.mean(edge_pixels)
        overall_std = np.std(heatmap)
        if center_intensity > edge_intensity * 1.5:
            return "center_focused"
        elif edge_intensity > center_intensity * 1.5:
            return "edge_focused"
        elif overall_std < 0.1:
            return "distributed"
        else:
            return "mixed_pattern"
    
    def _generate_text_explanation(self, predicted_class: int, confidence: float, 
                                 region_analysis: Dict, attention_stats: Dict) -> str:
        prediction = self.class_names[predicted_class]
        explanation = f"Kết quả: Hình ảnh được phân loại là '{prediction}' với độ tin cậy {confidence*100:.1f}%.\n\n"
        explanation += "Căn cứ phân tích:\n"
        top_regions = region_analysis['top_regions']
        focus_pattern = region_analysis['focus_distribution']
        region_explanations = {
            'top_center': 'logo ',
            'center': 'phần chính của sản phẩm (chất liệu hoặc hoa văn)',
            'edges': 'đường viền (kiểm tra chất lượng may hoặc in)',
            'top_left': 'góc trên trái (thường chứa watermark hoặc ký hiệu nhỏ)',
            'top_right': 'góc trên phải (thường chứa nhãn phụ)',
            'bottom_center': 'thông tin in phía dưới (serial number hoặc nhãn thương hiệu)',
            'center_left': 'chi tiết bên trái (có thể là khóa kéo hoặc nút)',
            'center_right': 'chi tiết bên phải (có thể là logo phụ)',
            'edges': 'các đường viền (kiểm tra độ hoàn thiện)'
        }
        for i, (region_name, region_data) in enumerate(top_regions):
            intensity = region_data['mean_intensity']
            coverage = region_data['coverage']
            if intensity > 0.1:
                explanation += f"{i+1}. Vùng {region_explanations.get(region_name, region_name)}: "
                explanation += f"Thu hút sự chú ý cao ({intensity*100:.1f}%), "
                explanation += f"với {coverage*100:.1f}% diện tích có độ quan trọng cao.\n"
        pattern_explanations = {
            'center_focused': 'Model tập trung vào logo hoặc chi tiết chính của sản phẩm.',
            'edge_focused': 'Model chú ý đến các cạnh, có thể liên quan đến chất lượng gia công.',
            'distributed': 'Model phân tích toàn bộ hình ảnh, kiểm tra nhiều đặc điểm.',
            'mixed_pattern': 'Model sử dụng kết hợp nhiều vùng để đưa ra quyết định.'
        }
        explanation += f"\nMô hình attention: {pattern_explanations.get(focus_pattern, focus_pattern)}\n"
        concentration = attention_stats['focus_concentration']
        if concentration > 3:
            explanation += "\nModel có sự tập trung cao vào một số vùng cụ thể, cho thấy có đặc điểm rõ ràng để phân biệt.\n"
        elif concentration < 1.5:
            explanation += "\nModel phân tán attention, có thể cần nhiều đặc điểm nhỏ để đưa ra quyết định.\n"
        if predicted_class == 1:  # Fake
            explanation += "\nCác dấu hiệu có thể chỉ ra sản phẩm giả:\n"
            explanation += "- Logo có sai lệch về hình dạng hoặc vị trí.\n"
            explanation += "- Màu sắc không đúng với tiêu chuẩn thương hiệu.\n"
            explanation += "- Chất lượng gia công kém (đường may không đều, chi tiết in mờ).\n"
        else:  # Real
            explanation += "\nCác dấu hiệu chỉ ra sản phẩm thật:\n"
            explanation += "- Logo đúng tỷ lệ và vị trí.\n"
            explanation += "- Màu sắc chuẩn theo thương hiệu.\n"
            explanation += "- Đường may và chi tiết sắc nét, đồng nhất.\n"
        return explanation
    
    def visualize_explanation(self, image: torch.Tensor, original_image: np.ndarray, 
                            explanation_result: Dict, save_path: str = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Ảnh gốc')
        axes[0, 0].axis('off')
        heatmap = explanation_result['attention_heatmap']
        im1 = axes[0, 1].imshow(heatmap, cmap='hot', alpha=0.8)
        axes[0, 1].set_title('Attention Heatmap')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        overlay = original_image.copy().astype(float) / 255.0
        heatmap_colored = plt.cm.hot(heatmap)[:, :, :3]
        blended = 0.6 * overlay + 0.4 * heatmap_colored
        axes[1, 0].imshow(blended)
        axes[1, 0].set_title('Overlay: Ảnh + Attention')
        axes[1, 0].axis('off')
        axes[1, 1].text(0.1, 0.9, explanation_result['explanation'], 
                       transform=axes[1, 1].transAxes, fontsize=10,
                       verticalalignment='top', wrap=True,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Giải thích')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return fig

def demo_explainable_prediction(image_path: str, checkpoint_path: str = '../saved_models/best_model_3.pth'):
    config = {
        'image_size': 224,
        'patch_size': 16,
        'num_classes': 2,
        'dim': 192,
        'depth': 4,
        'heads': 6,
        'mlp_dim': 384
    }
    model = ExplainableVisionTransformer(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim']
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} không tồn tại. Vui lòng huấn luyện mô hình trước.")
        return
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((config['image_size'], config['image_size']))
        original_image = np.array(image_resized)
        image_tensor = transform(image).to(device)
    except FileNotFoundError:
        print(f"Ảnh {image_path} không tồn tại.")
        return
    analyzer = ExplainabilityAnalyzer(model, class_names=['Real', 'Fake'])
    result = analyzer.predict_with_explanation(image_tensor, original_image)
    print("=== KẾT QUẢ PHÂN TÍCH ===")
    print(f"Dự đoán: {result['prediction']}")
    print(f"Độ tin cậy: {result['confidence']*100:.1f}%")
    print("\n=== GIẢI THÍCH ===")
    print(result['explanation'])
    save_path = f"results/predictions/prediction_{os.path.basename(image_path)}.png"
    fig = analyzer.visualize_explanation(image_tensor, original_image, result, save_path)
    plt.show()
    return result

if __name__ == "__main__":
    image_path = input("Nhập đường dẫn đến ảnh cần dự đoán: ")
    if os.path.exists(image_path):
        demo_explainable_prediction(image_path)
    else:
        print(f"Ảnh {image_path} không tồn tại")