# KL-1 Authentic Check AI

🎯 **Mục tiêu**: Phân loại hàng thật/giả với độ chính xác 85-90%

## 🚀 Sử dụng

### 1. Cài đặt
```bash
pip install -r config/requirements.txt
```

### 2. Training
```bash
cd src
python train.py
```

### 3. Chạy Web + API
```bash
cd src
python api.py
```
- Web Interface: http://127.0.0.1:8000
- API Docs: http://127.0.0.1:8000/docs

## 📁 Structure
```
KL-1/
├── src/
│   ├── train.py              # Training chính
│   ├── api.py                # Web + API server
│   ├── explainable_model.py  # Model với AI explanation
│   └── data/                 # Dataset
├── config/
│   ├── config.yaml          # Cấu hình training
│   └── requirements.txt     # Dependencies
└── web/
    └── index.html           # Giao diện web
```

## 🎯 Tính năng

- **Explainable AI**: Giải thích vì sao AI đưa ra quyết định
- **Attention Visualization**: Hiển thị vùng AI tập trung phân tích
- **Web Interface**: Giao diện đẹp, dễ sử dụng
- **High Accuracy**: Tối ưu cho 85-90% accuracy
- **Fast Training**: Balanced speed/accuracy

## 📊 Dataset

Chuẩn bị dữ liệu:
```
src/data/
├── train/
│   ├── real/     # Ảnh thật
│   └── fake/     # Ảnh giả
└── validation/
    ├── real/     # Ảnh thật (validation)
    └── fake/     # Ảnh giả (validation)
```

## 🔧 API Usage

```python
import requests
import base64

# Encode image
with open('image.jpg', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Predict
response = requests.post('http://127.0.0.1:8000/api/predict', 
                        json={'image': img_base64})
result = response.json()

print(f"Prediction: {result['result']['class']}")
print(f"Confidence: {result['result']['confidence']}%")
print(f"Explanation: {result['result']['explanation']}")
``` 