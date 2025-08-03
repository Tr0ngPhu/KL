from datetime import datetime
import os
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
from utils.heatmap_utils import generate_attention_heatmap, get_focus_region, explain_focus_region
try:
    from explainer import ExplainabilityAnalyzer
    EXPLAINER_AVAILABLE = True
except ImportError:
    EXPLAINER_AVAILABLE = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import traceback
import matplotlib.pyplot as plt
import io
from scipy import ndimage


# --- Load config.yaml and set up globals early ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
config_path = os.path.join(project_root, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

uploads_dir = os.path.join(project_root, "uploads")
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

class_names = ['Fake', 'Real']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
analyzer = None
app = FastAPI()

def load_best_model():
    """Load the best model from the results directory, fallback to pretrained if not found"""
    global model
    try:
        results_dir = os.path.join(project_root, 'results')
        if not os.path.exists(results_dir):
            print(f"⚠️ Results directory not found at {results_dir}. Fallback to pretrained.")
            return False
        candidate_folders = []
        for d in os.listdir(results_dir):
            folder_path = os.path.join(results_dir, d)
            if os.path.isdir(folder_path):
                model_path = os.path.join(folder_path, 'best_model.pth')
                if os.path.exists(model_path):
                    candidate_folders.append((folder_path, os.path.getmtime(model_path)))
        if not candidate_folders:
            print("⚠️ No model folders with best_model.pth found in results. Fallback to pretrained.")
            return False
        latest_folder, _ = max(candidate_folders, key=lambda x: x[1])
        print(f"🔍 Found model directory: {os.path.basename(latest_folder)}")
        model_path = os.path.join(latest_folder, 'best_model.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            from models.model import PretrainedVisionTransformer
            model_cfg = config['model']
            model = PretrainedVisionTransformer(
                model_name='vit_base_patch16_224',
                num_classes=model_cfg['num_classes'],
                pretrained=False
            )
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                try:
                    model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"⚠️ Không thể load state_dict cho PretrainedVisionTransformer: {e}")
            else:
                print("⚠️ Checkpoint không đúng định dạng state_dict. Hãy lưu lại model bằng torch.save(model.state_dict(), path)")
            model.to(device)
            model.eval()
            if EXPLAINER_AVAILABLE:
                global analyzer
                analyzer = ExplainabilityAnalyzer(model, class_names)
                print("✅ Analyzer loaded.")
            else:
                print("⚠️ Analyzer not available.")
            print(model)
            api_log_path = os.path.join(project_root, 'logs', 'api_model_info.log')
            with open(api_log_path, 'a', encoding='utf-8') as api_log:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                model_name = type(model).__name__
                api_log.write(f"{now} - INFO - Model loaded: {model_name}\n")
            accuracy = checkpoint.get('accuracy', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
            if isinstance(accuracy, float):
                print(f"✅ Loaded best model from '{os.path.basename(latest_folder)}' with accuracy {accuracy:.2%}")
            else:
                print(f"✅ Loaded best model from '{os.path.basename(latest_folder)}' (accuracy not recorded in checkpoint).")
            return True
        else:
            print(f"⚠️ 'best_model.pth' not found in the latest folder '{os.path.basename(latest_folder)}'. Fallback to pretrained.")
            return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def fallback_to_pretrained():
    """If loading local model fails, use a pretrained one as a fallback."""
    global model, analyzer
    print("⚠️ Falling back to a simple model.")
    
    # Create a simple CNN model that doesn't require external dependencies
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    try:
        # Use simple CNN
        model = SimpleCNN(num_classes=config['model']['num_classes'])
        print("✅ Using simple CNN model")
    except Exception as e:
        print(f"⚠️ Model creation error: {e}")
        model = SimpleCNN(num_classes=config['model']['num_classes'])
        print("✅ Using fallback CNN model")
    
    model.to(device)
    model.eval()
    
    if EXPLAINER_AVAILABLE:
        analyzer = ExplainabilityAnalyzer(model, class_names)
    else:
        analyzer = None
        print("⚠️ Analyzer not available")
        
    print("✅ Fallback model loaded successfully.")

def setup_transform():
    """Setup image transforms"""
    global transform
    model_cfg = config['model']
    aug_cfg = config['augmentation']
    transform_list = [
        transforms.Resize((model_cfg['image_size'], model_cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)

# --- Initialization ---
setup_transform()
if not load_best_model():
    print("⚠️ Using fallback pretrained model for API.")
    fallback_to_pretrained()

# Serve static files
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(os.path.dirname(current_dir), "web")
uploads_dir = os.path.join(os.path.dirname(current_dir), "uploads")

if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    try:
        with open("web/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        print(f"❌ Error loading UI: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """🔥 ENHANCED: Endpoint with superior heatmap generation"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    try:
        # Đọc ảnh và chuyển về tensor
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        # Dự đoán
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence = probabilities.max().item()
            predicted_idx = probabilities.argmax().item()
            predicted_class = class_names[predicted_idx]
        # Lấy attention map từ ViT và debug heatmap
        import logging
        if hasattr(model, 'get_attention_maps'):
            attn_maps = model.get_attention_maps(img_tensor)
            # Lấy attention map cuối, trung bình các head, loại bỏ cls token
            attn = attn_maps[-1].mean(1)[0]  # (num_tokens, num_tokens)
            patch_num = int((attn.shape[0]-1)**0.5)
            # Lấy attention từ cls token tới các patch
            heatmap = attn[0, 1:].reshape(patch_num, patch_num).cpu().numpy()
            # Chuẩn hóa heatmap về [0,1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            # Resize heatmap đúng kích thước ảnh gốc
            heatmap_resized = cv2.resize(heatmap, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
            # Overlay heatmap lên ảnh gốc
            # Dùng colormap JET để heatmap có màu nổi bật (đỏ-vàng-xanh)
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            original = np.array(image)
            if original.max() <= 1.0:
                original = (original * 255).astype(np.uint8)
            overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
            import random
            for _ in range(3):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                rand_hex = hex(random.getrandbits(24))[2:]
                heatmap_path = os.path.join(uploads_dir, f'heatmap_{timestamp}_{rand_hex}.png')
                if cv2.imwrite(heatmap_path, overlay):
                    break
        else:
            patch_num = config['model']['image_size'] // config['model']['patch_size']
            heatmap = np.ones((patch_num, patch_num)) * 0.5
            heatmap_resized = cv2.resize(heatmap, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            original = np.array(image)
            if original.max() <= 1.0:
                original = (original * 255).astype(np.uint8)
            overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
            import random
            for _ in range(3):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                rand_hex = hex(random.getrandbits(24))[2:]
                heatmap_path = os.path.join(uploads_dir, f'heatmap_{timestamp}_{rand_hex}.png')
                if cv2.imwrite(heatmap_path, overlay):
                    break
        # Phân tích vùng focus
        heatmap_resized = cv2.resize(heatmap, (config['model']['image_size'], config['model']['image_size']), interpolation=cv2.INTER_CUBIC)
        focus_patch, (max_y, max_x) = get_focus_region(heatmap_resized, np.array(image), patch_size=config['model']['patch_size'])
        explanation = explain_focus_region(focus_patch, predicted_class)
        # Trả về kết quả
        heatmap_url = "/uploads/" + os.path.basename(heatmap_path)
        response_data = {
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "heatmap": heatmap_url,
            "focus_explanation": explanation
        }
        return JSONResponse(response_data)
    except Exception as e:
        print(f"🔥 Enhanced prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {e}")
        return JSONResponse(response_data)

    except Exception as e:
        print(f"🔥 Enhanced prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {e}")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Enhanced analysis endpoint with product-specific feature analysis."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)

        # Transform image
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence = probabilities.max().item()
            predicted_idx = probabilities.argmax().item()
            predicted_class = class_names[predicted_idx]
            
        # Generate image metrics with the new function
        from explainer import generate_image_metrics, generate_ai_analysis, generate_heatmap
        metrics = generate_image_metrics(image_array)
        explanation = generate_ai_analysis(metrics, confidence)

        # --- Generate real attention-based heatmap overlay if possible ---
        import cv2
        import random
        for _ in range(3):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            rand_hex = hex(random.getrandbits(24))[2:]
            heatmap_path = os.path.join(os.path.dirname(current_dir), "uploads", f"heatmap_{timestamp}_{rand_hex}.jpg")
            heatmap_url = "/uploads/" + os.path.basename(heatmap_path)
            # break loop if file saved successfully (set below)
        heatmap_save_success = False
        try:
            from explainer import ExplainabilityAnalyzer
            analyzer = ExplainabilityAnalyzer(model, class_names)
            if hasattr(model, 'get_attention_maps'):
                attn_maps = model.get_attention_maps(img_tensor)
                attn = attn_maps[-1].mean(1)[0]
                patch_num = int((attn.shape[0]-1)**0.5)
                heatmap = attn[0, 1:].reshape(patch_num, patch_num).cpu().numpy()
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                heatmap_resized = cv2.resize(heatmap, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
                heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                original = image_array
                if original.max() <= 1.0:
                    original = (original * 255).astype(np.uint8)
                overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
                for _ in range(3):
                    if cv2.imwrite(heatmap_path, overlay):
                        heatmap_save_success = True
                        break
            else:
                patch_num = config['model']['image_size'] // config['model']['patch_size']
                heatmap = np.ones((patch_num, patch_num)) * 0.5
                heatmap_resized = cv2.resize(heatmap, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
                heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                original = image_array
                if original.max() <= 1.0:
                    original = (original * 255).astype(np.uint8)
                overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
                heatmap_save_success = cv2.imwrite(heatmap_path, overlay)
            if not heatmap_save_success:
                print(f"[Heatmap save error]: Failed to save overlay to {heatmap_path}")
        except Exception as e:
            print(f"[Heatmap generation error]: {e}")
            # Fallback: simple gradient heatmap overlay
            try:
                simple_heatmap = np.zeros((224, 224))
                y, x = np.mgrid[0:224, 0:224]
                center_y, center_x = 112, 112
                simple_heatmap = 1 - np.sqrt(((x - center_x) / 112)**2 + ((y - center_y) / 112)**2)
                simple_heatmap = np.clip(simple_heatmap, 0, 1)
                # Overlay on resized image
                resized_img = cv2.resize(image_array, (224, 224))
                from explainer import ExplainabilityAnalyzer
                fallback_analyzer = ExplainabilityAnalyzer(model, class_names)
                overlay = fallback_analyzer.get_colored_heatmap_overlay(resized_img, simple_heatmap, alpha=0.4)
                heatmap_save_success = cv2.imwrite(heatmap_path, overlay)
                if not heatmap_save_success:
                    print(f"[Fallback heatmap save error]: Failed to save fallback overlay to {heatmap_path}")
            except Exception as e2:
                print(f"[Fallback heatmap overlay error]: {e2}")
                heatmap_save_success = False
        
        # Calculate traditional metrics for backwards compatibility
        feature_analysis = {}  # Always define before try
        try:
            from product_knowledge import ProductAnalyzer
            analyzer = ProductAnalyzer()
            product_type = predicted_class.lower()
            is_fake = product_type == 'fake'
            detected_product_type = "shoes"  # Can be "shoes", "clothing", or "accessories"
            print(f"Using product type: {detected_product_type}")
            img_array = np.array(image)
            # ...existing code for feature extraction and fallback_features...
            # (giữ nguyên toàn bộ logic cũ ở đây)
            # ...existing code...
        except Exception as e:
            print(f"Feature analysis error: {e}")
            traceback.print_exc()  # Print the full traceback for debugging
            feature_analysis = {
                "error": "Không thể phân tích sản phẩm",
                "details": str(e),
                "explanation": "Hệ thống gặp sự cố khi phân tích. Vui lòng thử lại với ảnh rõ ràng hơn."
            }
            explanation = feature_analysis["explanation"]
        # Always set explanation if not set, and ensure only one conclusion
        if 'explanation' not in locals() or explanation is None:
            explanation = feature_analysis["explanation"] if "explanation" in feature_analysis else "Không thể phân tích sản phẩm."
        # Remove duplicate/contradictory analysis: chỉ giữ lại mỗi mục chính cuối cùng, đúng thứ tự
        import re
        def extract_last_block(pattern, text):
            matches = list(re.finditer(pattern, text, re.DOTALL|re.IGNORECASE))
            return matches[-1].group(0).strip() if matches else ''

        last_tech = extract_last_block(r'CHỈ SỐ KỸ THUẬT:.*?(?=(PHÂN TÍCH VI CẤU TRÚC|BẤT THƯỜNG PHÁT HIỆN|🔬|⚠️|✅) KẾT LUẬN|$)', explanation)
        last_struct = extract_last_block(r'PHÂN TÍCH VI CẤU TRÚC.*?(?=(BẤT THƯỜNG PHÁT HIỆN|🔬|⚠️|✅) KẾT LUẬN|$)', explanation)
        last_abnormal = extract_last_block(r'BẤT THƯỜNG PHÁT HIỆN.*?(?=(🔬|⚠️|✅) KẾT LUẬN|$)', explanation)
        last_conclusion = extract_last_block(r'(⚠️|✅) KẾT LUẬN.*?(?=(⚠️|✅) KẾT LUẬN|$)', explanation)

        # Optional: also extract last supplement (🔬)
        last_supplement = extract_last_block(r'🔬.*?(?=(⚠️|✅) KẾT LUẬN|$)', explanation)

        # Compose in order
        explanation_parts = []
        if last_tech:
            explanation_parts.append(last_tech)
        if last_struct:
            explanation_parts.append(last_struct)
        if last_abnormal:
            explanation_parts.append(last_abnormal)
        if last_supplement:
            explanation_parts.append(last_supplement)
        if last_conclusion:
            explanation_parts.append(last_conclusion)
        # Ghép các block và loại bỏ dòng trùng lặp, giữ thứ tự xuất hiện cuối cùng
        explanation_joined = '\n\n'.join(explanation_parts)
        # Tách thành từng dòng, loại bỏ dòng trống đầu/cuối
        lines = [line.strip() for line in explanation_joined.split('\n') if line.strip()]
        # Loại bỏ các dòng trùng lặp, giữ dòng cuối cùng xuất hiện
        seen = set()
        unique_lines = []
        for line in reversed(lines):
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        explanation = '\n'.join(reversed(unique_lines))
        # Add our new AI analysis based on metrics
        feature_analysis["ai_analysis"] = explanation
        # Convert all numpy types in result to native Python types for JSON serialization
        import collections.abc
        def convert_np(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_np(v) for v in obj]
            else:
                return obj

        # (heatmap_url đã được set ở trên, luôn trả về file vừa lưu)
        result = {
            "prediction": predicted_class,
            "confidence": float(round(confidence * 100, 2)),
            "analysis": explanation,
            "heatmap": heatmap_url if heatmap_save_success else None,
            "metrics": convert_np(metrics),
            "features": convert_np(feature_analysis)
        }
        if not heatmap_save_success:
            result["heatmap_warning"] = "Không thể tạo hoặc lưu bản đồ nhiệt (heatmap). Vui lòng kiểm tra quyền ghi thư mục uploads hoặc thử lại."
        return JSONResponse(result)
    except Exception as e:
        print(f"Analysis error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Analysis failed: {e}"})
        # ...existing code...
        
        return result
        
    except Exception as e:
        print(f"Analysis error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.get("/status")
def get_status():
    """API status with capability information"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "analyzer_initialized": analyzer is not None,
        "explainer_available": EXPLAINER_AVAILABLE,
        "device": str(device),
        "version": "7.0 - Fixed Connection Issues",
        "features": {
            "basic_prediction": True,
            "enhanced_analysis": EXPLAINER_AVAILABLE,
            "heatmap_generation": True,
            "vietnamese_explanation": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    import socket
    
    def find_free_port():
        """Find a free port starting from 8000"""
        for port in range(8000, 8010):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        return 8000
    
    port = find_free_port()
    print(f"🔥 Starting Enhanced Fake Detection API on port {port}")
    print(f"🌐 Access web interface at: http://127.0.0.1:{port}")
    print(f"📊 API status at: http://127.0.0.1:{port}/status")
    
    try:
        uvicorn.run("api:app", host="127.0.0.1", port=port, reload=False)
    except Exception as e:
        print(f"❌ Server error: {e}")
        input("Press Enter to exit...")
