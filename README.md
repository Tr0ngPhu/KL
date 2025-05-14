# Mô hình Phân loại Sản phẩm Thật/Giả

Dự án này sử dụng mô hình EfficientNet-B0 để phân loại hình ảnh sản phẩm thật/giả.

## Cấu trúc thư mục

```
.
├── data/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
├── models/
├── results/
├── model.py
├── train.py
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