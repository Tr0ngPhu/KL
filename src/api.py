import os
import torch
import torch.nn.functional as F
from torch.amp import autocast
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torchvision.transforms as transforms
import logging
import yaml
from datetime import datetime
import base64
from pydantic import BaseModel
from typing import Optional

# Thiết lập
current_dir = os.path.dirname(os.path.abspath(__file__))
from models.model import VisionTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = FastAPI(title="KL Fake Detection API", version="2.0")

# Models
class Base64ImageRequest(BaseModel):
    image: str
    
class PredictionResponse(BaseModel):
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None

# Biến toàn cục
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
transform = None
class_names = ['Giả', 'Thật']

logging.info(f'API đang chạy trên: {device}')

def load_config():
    """Load cấu hình từ file config.yaml"""
    config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_transforms():
    """Thiết lập transforms cho inference"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def find_latest_checkpoint():
    """Tìm checkpoint mới nhất"""
    results_dir = os.path.join(os.path.dirname(current_dir), 'results')
    if not os.path.exists(results_dir):
        return None
    
    timestamp_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            model_file = os.path.join(item_path, 'best_model.pth')
            if os.path.exists(model_file):
                timestamp_dirs.append((item, model_file))
    
    if not timestamp_dirs:
        return None
    
    timestamp_dirs.sort(key=lambda x: x[0], reverse=True)
    return timestamp_dirs[0][1]

def load_model():
    """Load model đã train từ checkpoint"""
    global model, transform
    
    try:
        model_config = load_config()
        
        model = VisionTransformer(
            image_size=model_config['model']['image_size'],
            patch_size=model_config['model']['patch_size'],
            num_classes=model_config['model']['num_classes'],
            dim=model_config['model']['dim'],
            depth=model_config['model']['depth'],
            heads=model_config['model']['heads'],
            mlp_dim=model_config['model']['mlp_dim']
        )
        
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f'Model được load từ: {checkpoint_path}')
            logging.info(f'Độ chính xác model: {checkpoint.get("val_acc", "N/A"):.2f}%')
        else:
            logging.warning('Không tìm thấy checkpoint, dùng model chưa train')
        
        model.to(device)
        model.eval()
        transform = setup_transforms()
        
        logging.info('Model đã load thành công!')
        return True
        
    except Exception as e:
        logging.error(f'Lỗi khi load model: {e}')
        return False

def allowed_file(filename):
    """Kiểm tra định dạng file có hợp lệ không"""
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def predict_image(image):
    """Dự đoán ảnh với model"""
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = predicted.cpu().item()
        confidence_score = confidence.cpu().item()
        
        result = {
            'is_real': bool(predicted_class == 1),
            'class': class_names[predicted_class],
            'confidence': round(confidence_score * 100, 2),
            'probabilities': {
                'fake': round(probabilities[0][0].cpu().item() * 100, 2),
                'real': round(probabilities[0][1].cpu().item() * 100, 2)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logging.info(f'Kết quả dự đoán: {result["class"]} ({result["confidence"]}%)')
        return result
        
    except Exception as e:
        logging.error(f'Lỗi khi dự đoán: {e}')
        raise

# Các routes
@app.get("/")
async def root():
    return {
        "message": "KL Fake Detection API",
        "version": "2.0",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Định dạng file không hợp lệ")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        result = predict_image(image)
        return PredictionResponse(success=True, result=result)
        
    except Exception as e:
        logging.error(f'Lỗi trong predict endpoint: {e}')
        raise HTTPException(status_code=500, detail=f'Lỗi xử lý: {str(e)}')

@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(request: Base64ImageRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load")
    
    try:
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        result = predict_image(image)
        return PredictionResponse(success=True, result=result)
        
    except Exception as e:
        logging.error(f'Lỗi trong predict_base64 endpoint: {e}')
        raise HTTPException(status_code=500, detail=f'Lỗi xử lý: {str(e)}')

@app.on_event("startup")
async def startup_event():
    if not load_model():
        raise RuntimeError('Không thể load model')

if __name__ == "__main__":
    import uvicorn
    logging.info('Đang khởi động KL Fake Detection API...')
    uvicorn.run(app, host="0.0.0.0", port=5000)