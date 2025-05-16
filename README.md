# Mô hình Phân loại Sản phẩm Thật/Giả

Dự án này sử dụng mô hình VisionTransformer để phân loại hình ảnh sản phẩm thật/giả.

## Cấu trúc thư mục

```
.
├──api/
├──config/
├──logs/
├── data/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
├── src/
├── tests/
├── requirements.txt
├── requirements-m1.txt
└── setup.py
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
pip install -r requirements.txt #Windows
```

```bash
pip install -r requirements-m1.txt #Mac M1,M2
```

## Chuẩn bị dữ liệu

1. Tạo cấu trúc thư mục như trên
2. Đặt ảnh sản phẩm thật vào thư mục `/data/dataset/train/real`
3. Đặt ảnh sản phẩm giả vào thư mục `/data/dataset/train/fake`

## Huấn luyện mô hình

```bash
Cd src
python train.py
```

Kết quả huấn luyện sẽ được lưu trong thư mục `results/` với timestamp.

## Kiểm tra mô hình

```bash
Cd tests

## 1. Đánh giá mô hình:
bash
python test.py --model_path path/to/model.pth --test_dir data/test

## Ví dụ: python test.py --model_path ../src/results/20250515_084134/best_model.pth  --test_dir ../data/dataset/test

## 2. Chạy benchmark hiệu năng:
bash
python test.py --model_path path/to/model.pth --benchmark

## Ví dụ: python test.py --model_path ../src/results/20250515_084134/best_model.pth --benchmark

## 3. Tuỳ chỉnh thiết bị:
bash
python test.py --model_path path/to/model.pth --device ## cuda / mps / cpu

## Ví dụ: python test.py --model_path ../src/results/20250515_084134/best_model.pth --device cuda

```

## Kiểm tra phân biệt trên Api

1.Kiểm tra trên terminal/cmd

```bash
    cd api
## Load Model Khởi động api

bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000

## Kiểm tra hình ảnh
bash
curl -X POST http://localhost:8000/predict \
  -F "file= path-file"

#  Ví dụ: curl -X POST http://localhost:8000/predict \
#   -F "file=@/Users/daidat/Documents/workspace/KL/data/dataset/test/fake/product_92c2b2ef.jpg"
```

## Tính năng

- Sử dụng VisionTransformer làm backbone
- Data augmentation để tăng cường dữ liệu
- Learning rate scheduling với CosineAnnealingWarmRestarts
- Gradient clipping để tránh exploding gradients
- Lưu model tốt nhất dựa trên độ chính xác validation
- Vẽ đồ thị loss và accuracy
- Tạo confusion matrix
- Báo cáo chi tiết về quá trình huấn luyện
