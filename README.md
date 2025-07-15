# KL - Phân loại hàng thật/giả với Vision Transformer

## Tổng quan
Dự án sử dụng Vision Transformer để phân loại hàng thật và hàng giả, được tối ưu hóa cho GPU RTX 3050 4GB.

## Các tính năng chính

### 🚀 Tối ưu hóa bộ nhớ
- **Mixed Precision Training**: Sử dụng FP16 để tiết kiệm 50% bộ nhớ
- **Gradient Accumulation**: Batch size nhỏ (4) nhưng effective batch size lớn (32)
- **Memory Monitoring**: Theo dõi usage GPU real-time
- **Optimized Model**: Giảm parameters từ 86M xuống ~22M

### 🎯 Tăng độ chính xác
- **Advanced Data Augmentation**: 
  - RandomCrop với resize lớn hơn
  - Perspective transformation
  - Gaussian blur simulation
  - Noise injection
- **Label Smoothing**: Giảm overfitting
- **OneCycle Learning Rate**: Tối ưu convergence
- **Weighted Sampling**: Xử lý class imbalance

### 🛠️ Cải tiến kỹ thuật
- **EarlyStopping**: Tự động dừng khi không cải thiện
- **Gradient Clipping**: Stable training
- **Clean Architecture**: Code dễ đọc, dễ maintain
- **Comprehensive Logging**: Theo dõi chi tiết quá trình training

## Cấu trúc dự án
```
KL-1/
├── config/
│   ├── config.yaml          # Cấu hình chính
│   └── requirements.txt     # Dependencies
├── src/
│   ├── data/
│   │   ├── dataset.py       # Custom dataset
│   │   ├── train/           # Dữ liệu training
│   │   ├── test/            # Dữ liệu test
│   │   └── validation/      # Dữ liệu validation
│   ├── models/
│   │   └── model.py         # Vision Transformer model
│   ├── utils/
│   │   └── utils.py         # Utility functions
│   └── train.py             # Training script (OPTIMIZED)
├── test_memory.py           # Kiểm tra memory usage
└── README.md               # Documentation
```

## Cải tiến so với version cũ

### 1. Memory Optimization
- **Trước**: Batch size 32, có thể OOM trên RTX 3050
- **Sau**: Batch size 4 + gradient accumulation = hiệu quả tương đương

### 2. Training Strategy
- **Trước**: ReduceLROnPlateau scheduler
- **Sau**: OneCycle scheduler + mixed precision

### 3. Data Augmentation
- **Trước**: Basic augmentation
- **Sau**: Advanced augmentation cho fake detection

### 4. Code Quality
- **Trước**: Monolithic training function
- **Sau**: Modular MemoryOptimizedTrainer class

## Hướng dẫn sử dụng

### 1. Cài đặt dependencies
```bash
# Kích hoạt virtual environment
source .venv/Scripts/activate

# Cài đặt packages
pip install -r config/requirements.txt
```

### 2. Chạy training
```bash
cd src
python train.py
```

### 3. Chạy API server
```bash
python api.py
```

### 4. Test API
```bash
python test_api.py
```

### 5. Theo dõi logs
```bash
tail -f logs/training.log
```

## API Endpoints

### 1. Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2025-07-15T10:30:00"
}
```

### 2. File Upload Prediction
```bash
POST /predict
Content-Type: multipart/form-data
```
Upload file với key `image`.

### 3. Base64 API Prediction
```bash
POST /api/predict
Content-Type: application/json
```
Request body:
```json
{
  "image": "base64_encoded_image_string"
}
```

Response:
```json
{
  "success": true,
  "result": {
    "is_real": true,
    "class": "Thật",
    "confidence": 87.5,
    "probabilities": {
      "fake": 12.5,
      "real": 87.5
    },
    "timestamp": "2025-07-15T10:30:00"
  }
}
```
└── requirements.txt
```

## Cài đặt

1. Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

1. Tạo cấu trúc thư mục như trên
2. Đặt ảnh sản phẩm thật vào thư mục `src/data/train/real`
3. Đặt ảnh sản phẩm giả vào thư mục `src/data/train/fake`

## Huấn luyện mô hình

```bash
Cd src
python train.py
```

Kết quả huấn luyện sẽ được lưu trong thư mục `results/` với timestamp.

## Tính năng

- Sử dụng EfficientNet-B0 làm backbone
- Data augmentation để tăng cường dữ liệu
- Learning rate scheduling với CosineAnnealingWarmRestarts
- Gradient clipping để tránh exploding gradients
- Lưu model tốt nhất dựa trên độ chính xác validation
- Vẽ đồ thị loss và accuracy
- Tạo confusion matrix
- Báo cáo chi tiết về quá trình huấn luyện 