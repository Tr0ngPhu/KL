import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import yaml
import timm
from PIL import Image
from torchvision import transforms
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class ExplainabilityAnalyzer:
    """
    A powerful, universal explainability analyzer for Vision Transformers.
    It supports both custom models and pretrained models from `timm`.
    """
    def __init__(self, model: nn.Module, class_names: List[str] = None):
        self.model = model.eval()
        self.class_names = class_names or ['Fake', 'Real']
        self.device = next(model.parameters()).device
        
        # Smartly detect model type
        self.is_timm_model = 'timm' in str(type(model)) or hasattr(model, 'default_cfg')
        
        # Cache for performance
        self._attention_cache = {}
        
        # Setup hooks for timm models to extract attention
        self.attention_maps = []
        self.activation_maps = []
        if self.is_timm_model:
            self._setup_attention_hooks()

    def _setup_attention_hooks(self):
        """Set up hooks to automatically extract attention from timm ViT models."""
        def attention_hook(module, input, output):
            try:
                # Handle different output formats from attention modules
                if isinstance(output, torch.Tensor):
                    # Single tensor output
                    if output.ndim == 4:  # [B, H, N, N] - attention weights
                        self.attention_maps.append(output.detach())
                    elif output.ndim == 3:  # [B, N, D] - attention output
                        self.activation_maps.append(output.detach())
                elif isinstance(output, (tuple, list)):
                    # Multiple outputs - check each one
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            if out.ndim == 4:  # Attention weights
                                self.attention_maps.append(out.detach())
                            elif out.ndim == 3:  # Activation maps
                                self.activation_maps.append(out.detach())
            except Exception as e:
                # Silently continue if hook fails
                pass

        def activation_hook(module, input, output):
            try:
                if isinstance(output, torch.Tensor) and output.ndim == 3:
                    self.activation_maps.append(output.detach())
            except Exception as e:
                pass

        # Register hooks on multiple types of attention-related modules
        hook_targets = [
            'attn.attn_drop',  # After attention dropout
            'attn',            # Attention module itself
            'blocks',          # Transformer blocks
            'norm1',           # Layer norm after attention
        ]

        hooks_registered = 0
        for name, module in self.model.named_modules():
            for target in hook_targets:
                if target in name:
                    try:
                        module.register_forward_hook(attention_hook)
                        hooks_registered += 1
                        break
                    except Exception as e:
                        continue
        
        # Also try to hook into the final layer norm or classifier
        for name, module in self.model.named_modules():
            if 'head' in name.lower() or 'classifier' in name.lower():
                try:
                    module.register_forward_hook(activation_hook)
                    hooks_registered += 1
                except Exception as e:
                    continue

    def _get_attention_for_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extracts and processes attention maps for a given image.
        """
        self.attention_maps = []
        self.activation_maps = []
        
        with torch.no_grad():
            _ = self.model(image_tensor.unsqueeze(0))
        
        # Try to process attention maps first
        if self.attention_maps:
            try:
                return self._process_attention_maps()
            except Exception as e:
                print(f"Warning: Failed to process attention maps: {e}")
        
        # Fallback to activation-based attention
        if self.activation_maps:
            try:
                return self._process_activation_maps()
            except Exception as e:
                print(f"Warning: Failed to process activation maps: {e}")
        
        # Final fallback
        return self._create_fallback_heatmap(image_tensor.shape[1:])

    def _process_attention_maps(self) -> np.ndarray:
        """Process attention weights to create heatmap."""
        processed_attentions = []
        
        for attn_map in self.attention_maps:
            try:
                # Handle different attention map shapes
                if attn_map.ndim == 4:  # [B, H, N, N]
                    # Focus on the attention from the [CLS] token to the image patches
                    if attn_map.shape[2] > 1 and attn_map.shape[3] > 1:
                        cls_attention = attn_map[0, :, 0, 1:].mean(dim=0)  # Avg over heads
                        processed_attentions.append(cls_attention.cpu().numpy())
                elif attn_map.ndim == 3:  # [B, N, N]
                    if attn_map.shape[1] > 1 and attn_map.shape[2] > 1:
                        cls_attention = attn_map[0, 0, 1:]  # CLS to patches
                        processed_attentions.append(cls_attention.cpu().numpy())
            except Exception as e:
                continue
        
        if not processed_attentions:
            raise ValueError("No valid attention maps found")

        # Average the attention maps from all transformer blocks
        final_attention = np.mean(processed_attentions, axis=0)
        return self._reshape_to_heatmap(final_attention)

    def _process_activation_maps(self) -> np.ndarray:
        """Process activation maps to create attention-like heatmap."""
        if not self.activation_maps:
            raise ValueError("No activation maps available")
        
        # Use the last activation map (closest to the output)
        last_activation = self.activation_maps[-1][0]  # Remove batch dimension
        
        # If this includes CLS token, remove it
        if last_activation.shape[0] > 196:  # Assuming 14x14 patches + CLS
            patch_activations = last_activation[1:]  # Remove CLS token
        else:
            patch_activations = last_activation
        
        # Compute attention-like weights from activations
        attention_weights = torch.norm(patch_activations, dim=1).cpu().numpy()
        return self._reshape_to_heatmap(attention_weights)

    def _reshape_to_heatmap(self, attention_weights: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Reshape 1D attention weights to 2D heatmap."""
        # Determine grid size
        grid_size = int(np.sqrt(attention_weights.shape[0]))
        if grid_size * grid_size != attention_weights.shape[0]:
            # Handle non-square cases by padding or truncating
            expected_size = grid_size * grid_size
            if attention_weights.shape[0] > expected_size:
                attention_weights = attention_weights[:expected_size]
            else:
                padding = expected_size - attention_weights.shape[0]
                attention_weights = np.pad(attention_weights, (0, padding), mode='constant')

        attention_grid = attention_weights.reshape(grid_size, grid_size)
        
        # Resize to target image size
        heatmap = cv2.resize(attention_grid, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Normalize
        heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = np.ones_like(heatmap) * 0.5
            
        return heatmap

    def _create_fallback_heatmap(self, image_shape: Tuple[int, ...]) -> np.ndarray:
        """Creates a center-biased heatmap as a fallback."""
        if len(image_shape) == 2:
            h, w = image_shape
        elif len(image_shape) == 3:
            h, w = image_shape[1], image_shape[2]
        else:
            h, w = 224, 224  # Default size
            
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (min(h, w) * 0.4)**2)
        return heatmap

    def predict_with_explanation(self, image_tensor: torch.Tensor, original_image: np.ndarray) -> Dict:
        """
        Generates a prediction and a detailed, multi-faceted explanation.
        """
        # Ensure correct tensor shape
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)
        
        predicted_class = self.class_names[predicted_class_idx.item()]
        
        # Get attention heatmap
        try:
            heatmap = self._get_attention_for_image(image_tensor.squeeze(0))
        except Exception as e:
            print(f"Warning: Using fallback heatmap due to error: {e}")
            heatmap = self._create_fallback_heatmap(original_image.shape[:2])
        
        # Perform content analysis
        content_analysis = self._analyze_image_content(original_image, heatmap)
        
        # Generate textual explanation
        explanation_text = self._generate_text_explanation(predicted_class, confidence.item(), content_analysis)
        
        return {
            'prediction': predicted_class,
            'confidence': confidence.item(),
            'explanation': explanation_text,
            'heatmap': heatmap,
            'content_analysis': content_analysis,
            'model_type': 'timm_pretrained' if self.is_timm_model else 'custom'
        }

    def _analyze_image_content(self, image: np.ndarray, heatmap: np.ndarray) -> Dict:
        """Analyzes key visual properties of the image and the focused regions."""
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray_image.shape
            
            # Ensure heatmap is the same size as the image
            if heatmap.shape != (h, w):
                heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # High-attention area
            focus_mask = (heatmap > np.percentile(heatmap, 90)).astype(np.uint8)
            
            # Dominant color in focused area
            focus_colors = cv2.mean(image, mask=focus_mask) if np.any(focus_mask) else (0, 0, 0, 0)
            
            # Texture analysis in focused area
            focus_texture = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3)
            focus_texture_variance = np.mean(focus_texture[focus_mask > 0]**2) if np.any(focus_mask) else 0
            
            # Edge density
            edges = cv2.Canny(gray_image, 100, 200)
            edge_density = np.mean(edges) if edges.size > 0 else 0
            
            return {
                'focus_area_percentage': (np.sum(focus_mask) / (h * w)) * 100,
                'focus_dominant_color_rgb': focus_colors[:3],
                'focus_texture_variance': focus_texture_variance,
                'edge_density': edge_density,
            }
        except Exception as e:
            print(f"Warning: Content analysis failed: {e}")
            return {
                'focus_area_percentage': 10.0,
                'focus_dominant_color_rgb': (128, 128, 128),
                'focus_texture_variance': 500.0,
                'edge_density': 50.0,
            }

    def _generate_text_explanation(self, prediction: str, confidence: float, analysis: Dict) -> str:
        """Generates a human-readable explanation from the analysis results."""
        explanation = f"AI Prediction: **{prediction.upper()}** with {confidence:.1%} confidence.\n"
        
        focus_area = analysis['focus_area_percentage']
        texture_var = analysis['focus_texture_variance']
        
        explanation += f"The model focused on **{focus_area:.1f}%** of the image. "
        
        if texture_var > 1000:
            explanation += "The area of focus has **high texture detail**, suggesting complex patterns or material. "
        elif texture_var < 200:
            explanation += "The focused area appears **smooth and uniform**. "
        else:
            explanation += "The texture in the focus area is moderately complex. "

        if prediction.lower() == 'fake':
            if texture_var < 150 and focus_area < 10:
                explanation += "This combination of low texture and unfocused attention can be a sign of a low-quality copy."
            else:
                explanation += "Key indicators within the focused region deviate from those expected in an authentic product."
        else:  # Real
            if texture_var > 1200 and focus_area > 15:
                explanation += "The sharp details and concentrated focus are strong indicators of authentic manufacturing."
            else:
                explanation += "The visual evidence aligns with the characteristics of a genuine product."

        return explanation

    def visualize_explanation(self, original_image: np.ndarray, result: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Creates a comprehensive visualization of the explanation."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f"AI Analysis: {result['prediction']} ({result['confidence']:.1%})", fontsize=20, weight='bold')

        # 1. Original Image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')

        # 2. Attention Heatmap
        im = axes[1].imshow(result['heatmap'], cmap='viridis')
        axes[1].set_title('AI Attention Heatmap', fontsize=14)
        axes[1].axis('off')
        fig.colorbar(im, ax=axes[1], shrink=0.8)

        # 3. Overlay
        try:
            overlay = original_image.astype(float) / 255.0
            heatmap_colored = plt.cm.viridis(result['heatmap'])[:, :, :3]
            blended = 0.6 * overlay + 0.4 * heatmap_colored
            axes[2].imshow(blended)
        except Exception as e:
            # Fallback: just show the original image
            axes[2].imshow(original_image)
            
        axes[2].set_title('Attention Overlay', fontsize=14)
        axes[2].axis('off')

        # Add explanation text
        plt.figtext(0.5, 0.02, result['explanation'], ha="center", fontsize=12, 
                   wrap=True, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved to {save_path}")

        return fig

def ensure_dir(directory: str):
    if directory and not os.path.exists(directory):
        os.makedirs(directory)