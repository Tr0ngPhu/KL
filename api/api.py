# api.py
import os
import sys
import yaml
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.explainable_model import ExplainableVisionTransformer, ExplainabilityAnalyzer
from src.utils import ensure_dir

app = FastAPI(
    title="API Phát Hiện Sản Phẩm Giả",
    description="API phát hiện sản phẩm giả sử dụng Explainable Vision Transformer"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)
MAX_FILE_SIZE = 10 * 1024 * 1024

logger.info(f"Tạo thư mục uploads: {UPLOAD_FOLDER}")
try:
    ensure_dir(UPLOAD_FOLDER)
except Exception as e:
    logger.error(f"Không thể tạo thư mục {UPLOAD_FOLDER}: {e}")
    sys.exit(1)

app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

def load_config():
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    logger.info(f"Đọc cấu hình từ: {config_path}")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Không tìm thấy config.yaml, sử dụng cấu hình mặc định")
        return {
            'model': {
                'image_size': 224,
                'patch_size': 16,
                'num_classes': 2,
                'dim': 192,
                'depth': 4,
                'heads': 6,
                'mlp_dim': 384
            },
            'paths': {
                'checkpoint_dir': '../saved_models'
            }
        }

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model():
    logger.info("Tải mô hình mặc định")
    try:
        config = load_config()
        device = get_device()
        model = ExplainableVisionTransformer(
            image_size=config['model']['image_size'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            dim=config['model']['dim'],
            depth=config['model']['depth'],
            heads=config['model']['heads'],
            mlp_dim=config['model']['mlp_dim']
        ).to(device)
        checkpoint_path = os.path.join(project_root, config['paths']['checkpoint_dir'], 'best_model_4.pth')
        logger.info(f"Tải checkpoint từ: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} không tồn tại")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        num_patches = (config['model']['image_size'] // config['model']['patch_size']) ** 2
        logger.info(f"num_patches: {num_patches}, grid_size: {int(np.sqrt(num_patches))}")
        return model
    except Exception as e:
        logger.error(f"Không thể tải mô hình: {str(e)}")
        raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def save_heatmap(analyzer, image_tensor, original_image, result, save_path):
    logger.info(f"Lưu heatmap vào: {save_path}")
    try:
        heatmap = result['attention_heatmap']
        logger.debug(f"Heatmap shape: {heatmap.shape}")
        if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
            logger.error(f"Heatmap không phải mảng 2D NumPy, type: {type(heatmap)}, shape: {getattr(heatmap, 'shape', 'N/A')}")
            raise ValueError(f"Heatmap phải là mảng 2D NumPy, nhận được type: {type(heatmap)}, shape: {getattr(heatmap, 'shape', 'N/A')}")
        
        if heatmap.shape[0] < 1 or heatmap.shape[1] < 1:
            logger.error(f"Kích thước heatmap không hợp lệ: {heatmap.shape}")
            raise ValueError(f"Kích thước heatmap không hợp lệ: {heatmap.shape}")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(original_image)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        logger.error(f"Không thể lưu heatmap: {str(e)}")
        raise

try:
    model = load_model()
    device = get_device()
    analyzer = ExplainabilityAnalyzer(model, class_names=['Real', 'Fake'])
    logger.info(f"Đã tải mô hình thành công trên thiết bị: {device}")
except Exception as e:
    logger.error(f"Không thể khởi tạo mô hình: {e}")
    sys.exit(1)

@app.get("/")
async def root():
    return {
        "message": "Chào mừng đến với API phát hiện sản phẩm giả",
        "hướng dẫn": "Gửi POST request đến /predict với file ảnh để nhận dự đoán và giải thích"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Nhận yêu cầu /predict với file: {file.filename}")
    if not file.filename or not allowed_file(file.filename):
        logger.error("Định dạng file không hợp lệ")
        raise HTTPException(
            status_code=400,
            detail="Định dạng file không được phép. Vui lòng upload ảnh JPG hoặc PNG."
        )
    
    if file.size > MAX_FILE_SIZE:
        logger.error(f"File quá lớn: {file.size} bytes")
        raise HTTPException(
            status_code=400,
            detail=f"File quá lớn. Kích thước tối đa là {MAX_FILE_SIZE / (1024*1024)}MB."
        )
    
    try:
        contents = await file.read()
        logger.info("Đã đọc nội dung file")
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_resized = image.resize(IMG_SIZE)
        original_image = np.array(image_resized)
        image_tensor = preprocess_image(image).to(device)
        logger.info("Đã tiền xử lý ảnh")
        
        result = analyzer.predict_with_explanation(image_tensor, original_image)
        logger.info(f"Kết quả từ predict_with_explanation: {result}")
        
        confidence = float(result['confidence'])
        prediction = result['prediction']
        if 0.4 < confidence < 0.9:  # Ngưỡng Suspicious
            prediction = "Suspicious"
            logger.info("Dự đoán được chuyển thành Suspicious do độ tin cậy trung bình")
        
        heatmap_path = os.path.join(UPLOAD_FOLDER, f"heatmap_{file.filename}.png")
        save_heatmap(analyzer, image_tensor, original_image, result, heatmap_path)
        
        risk_factors = []
        explanation_parts = result['explanation'].split('\n\n')
        for part in explanation_parts:
            if 'dấu hiệu' in part.lower():
                risk_factors = [line[2:] for line in part.split('\n') if line.startswith('-')]
        
        logger.info("Trả về kết quả thành công")
        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "Fake": round(float(result['probabilities'][1]) * 100, 2),
                "Real": round(float(result['probabilities'][0]) * 100, 2)
            },
            "explanation": result['explanation'],
            "riskFactors": risk_factors,
            "region_analysis": {
                "top_regions": [
                    {
                        "region_name": region[0],
                        "mean_intensity": round(float(region[1]['mean_intensity'] * 100), 2),
                        "coverage": round(float(region[1]['coverage'] * 100), 2)
                    }
                    for region in result['region_analysis']['top_regions']
                ],
                "focus_distribution": result['region_analysis']['focus_distribution']
            },
            "heatmap_path": f"/uploads/heatmap_{file.filename}.png"
        }
    
    except Exception as e:
        logger.error(f"Lỗi trong /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        log_level="info"
    )