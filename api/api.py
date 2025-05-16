import os
import sys
import yaml
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torchvision.transforms as transforms

# Thêm đường dẫn gốc vào sys.path để có thể import các module tự định nghĩa
current_dir = os.path.dirname(os.path.abspath(__file__))  # Lấy thư mục hiện tại
project_root = os.path.dirname(current_dir)  # Lấy thư mục gốc project
sys.path.insert(0, project_root)  # Thêm vào sys.path để import

from src.model import VisionTransformer  # Import model VisionTransformer tự định nghĩa
from src.utils import load_checkpoint, ensure_dir  # Import các hàm tiện ích

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Fake Product Detection API",
    description="API phát hiện sản phẩm giả sử dụng Vision Transformer"
)

# Cấu hình
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')  # Thư mục lưu file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Các định dạng file cho phép
IMG_SIZE = (224, 224)  # Kích thước ảnh đầu vào

# Tạo thư mục upload nếu chưa tồn tại
ensure_dir(UPLOAD_FOLDER)

def load_config():
    """Đọc file config.yaml và trả về dictionary cấu hình"""
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_device():
    """Xác định thiết bị chạy model (MPS/GPU/CPU)"""
    if torch.backends.mps.is_available():  # Kiểm tra Apple Silicon
        return torch.device("mps")
    elif torch.cuda.is_available():  # Kiểm tra GPU NVIDIA
        return torch.device("cuda")
    else:  # Fallback về CPU
        return torch.device("cpu")

def load_model():
    """Khởi tạo và load weights cho model"""
    try:
        config = load_config()  # Đọc cấu hình
        device = get_device()  # Xác định thiết bị
        
        # Khởi tạo model Vision Transformer
        model = VisionTransformer(
            image_size=config['model']['image_size'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            dim=config['model']['dim'],
            depth=config['model']['depth'],
            heads=config['model']['heads'],
            mlp_dim=config['model']['mlp_dim']
        ).to(device)  # Chuyển model sang device tương ứng
        
        # Đường dẫn đến file checkpoint
        checkpoint_path = os.path.join(
            project_root, 
            config['paths']['checkpoint_dir'], 
            'best_model.pth'
        )
        
        # Load weights từ checkpoint
        model = load_checkpoint(model=model, path=checkpoint_path)
        model.eval()  # Chuyển model sang chế độ evaluation
        return model
    except Exception as e:
        print(f"[LỖI] Không thể load model: {e}")
        raise

def allowed_file(filename):
    """Kiểm tra định dạng file có hợp lệ không"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image, device):
    """Tiền xử lý ảnh đầu vào cho model"""
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),  # Resize về kích thước chuẩn
        transforms.ToTensor(),  # Chuyển thành tensor
        # Chuẩn hóa theo mean và std của ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)  # Thêm chiều batch và chuyển sang device

# Khởi tạo model khi chạy ứng dụng
try:
    model = load_model()
    device = get_device()
    print(f"[THÔNG BÁO] Đã load model thành công trên thiết bị: {device}")
except Exception as e:
    print(f"[LỖI NGHIÊM TRỌNG] Không thể khởi tạo model: {e}")
    sys.exit(1)  # Thoát nếu không load được model

@app.get("/")
async def root():
    """Endpoint kiểm tra hoạt động của API"""
    return {
        "message": "Chào mừng đến với API phát hiện sản phẩm giả",
        "hướng dẫn": "Gửi POST request đến /predict với file ảnh để nhận dự đoán"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint nhận diện sản phẩm thật/giả
    
    Parameters:
    - file: File ảnh cần kiểm tra (JPG/PNG)
    
    Returns:
    - JSON chứa kết quả dự đoán và độ tin cậy
    """
    if not file.filename or not allowed_file(file.filename):
        return JSONResponse(
            status_code=400,
            content={"error": "Định dạng file không hỗ trợ. Vui lòng upload ảnh JPG hoặc PNG."}
        )
    
    try:
        # Đọc nội dung file
        contents = await file.read()
        
        # Mở ảnh và chuyển sang RGB (phòng trường hợp ảnh grayscale)
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Tiền xử lý ảnh
        image_tensor = preprocess_image(image, device)
        
        # Dự đoán
        with torch.no_grad():  # Tắt tính gradient để tiết kiệm bộ nhớ
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)  # Chuyển sang xác suất
            prediction = torch.argmax(probabilities, dim=1)  # Lấy class có xác suất cao nhất
            confidence = probabilities[0][prediction].item()  # Độ tin cậy
        
        # Chuyển kết quả thành nhãn
        predicted_label = 'Thật' if prediction.item() == 1 else 'Giả'
        
        # Trả về kết quả
        return {
            "prediction": predicted_label,
            "confidence": round(confidence * 100, 2),  # Làm tròn 2 chữ số
            "probabilities": {
                "Giả": round(probabilities[0][0].item() * 100, 2),
                "Thật": round(probabilities[0][1].item() * 100, 2)
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Có lỗi xảy ra: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    # Chạy server FastAPI
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Lắng nghe tất cả địa chỉ IP
        port=5000,  # Cổng 5000
        log_level="info"  # Hiển thị log
    )