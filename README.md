Mô hình Phân loại Sản phẩm Thật/Giả
Dự án này sử dụng mô hình VisionTransformer để phân loại hình ảnh sản phẩm thật/giả với khả năng giải thích (Explainable AI).
Cấu trúc thư mục
.
├── api/              # Chứa mã API FastAPI
├── config/           # Chứa file cấu hình (config.yaml)
├── logs/             # Lưu log huấn luyện và kiểm thử
├── data/             # Thư mục chứa dữ liệu
│   ├── dataset/
│   │   ├── train/    # Dữ liệu huấn luyện
│   │   │   ├── real/ # Ảnh sản phẩm thật
│   │   │   └── fake/ # Ảnh sản phẩm giả
│   │   ├── validation/ # Dữ liệu validation
│   │   └── test/     # Dữ liệu kiểm thử
│   │       ├── real/
│   │       └── fake/
├── src/              # Chứa mã nguồn (model, utils, train, test)
├── tests/            # Chứa script kiểm thử
├── uploads/          # Lưu heatmap từ API
├── requirements.txt  # Danh sách thư viện (Windows)
├── requirements-m1.txt # Danh sách thư viện (Mac M1/M2)
├── .gitignore        # File bỏ qua cho Git
├── README.md         # Tài liệu hướng dẫn
└── setup.py          # File setup cho gói Python

Cài đặt

Tạo môi trường ảo (khuyến nghị):
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Cài đặt các thư viện cần thiết:

Đối với Windows:pip install -r requirements.txt


Đối với Mac M1/M2:pip install -r requirements.txt





Chuẩn bị dữ liệu

Tạo cấu trúc thư mục như mô tả ở trên.

Đặt ảnh sản phẩm:

Ảnh thật vào /data/dataset/train/real/
Ảnh giả vào /data/dataset/train/fake/
Ảnh validation vào /data/dataset/validation/real/ và /data/dataset/validation/fake/
Ảnh kiểm thử vào /data/dataset/test/real/ và /data/dataset/test/fake/


Đảm bảo file config/config.yaml được cấu hình đúng (tham khảo file mẫu trong config/).


Hướng dẫn sử dụng
1. Huấn luyện mô hình

Chuyển đến thư mục src:cd src


Chạy script huấn luyện: python train.py


Kết quả (model, log, đồ thị) sẽ được lưu trong thư mục results/ với timestamp (ví dụ: results/20250713_224500/).

2. Kiểm tra mô hình

Chuyển đến thư mục tests:cd tests



a. Đánh giá mô hình

Chạy với đường dẫn model và thư mục test:

 python test.py --model_path path/to/model.pth --test_dir ../data/dataset/test


Ví dụ:python test.py --model_path ../src/results/20250713_224500/best_model.pth --test_dir ../data/dataset/test



b. Chạy benchmark hiệu năng

Đánh giá hiệu năng với các kích thước batch:
python test.py --model_path path/to/model.pth --benchmark


Ví dụ:python test.py --model_path ../src/results/20250713_224500/best_model.pth --benchmark



c. Tuỳ chỉnh thiết bị

Chọn thiết bị (cuda, mps, cpu):
python test.py --model_path path/to/model.pth --device [cuda/mps/cpu]


Ví dụ:python test.py --model_path ../src/results/20250713_224500/best_model.pth --device cuda



d. Dự đoán cho ảnh đơn lẻ

Chạy với đường dẫn ảnh:
python test.py --model_path path/to/model.pth --image_path path/to/image.jpg


Ví dụ:python test.py --model_path ../src/results/20250713_224500/best_model.pth --image_path ../data/dataset/test/real/product_001.jpg



3. Sử dụng API

Chuyển đến thư mục api:
cd api



a. Khởi động API

Chạy API trên terminal: python api.py



b. Gửi yêu cầu dự đoán

Sử dụng curl để kiểm tra ảnh:
curl -X POST http://localhost:5001/predict \
  -F "file=@path/to/image.jpg"


Ví dụ:
curl -X POST http://localhost:5001/predict \
  -F "file=@../data/dataset/test/fake/product_92c2b2ef.jpg"


Kết quả: API trả về JSON chứa dự đoán, độ tin cậy, giải thích, và đường dẫn heatmap.


Tính năng

Sử dụng VisionTransformer làm backbone.
Tích hợp ExplainabilityAnalyzer để phân tích vùng quan trọng.
Data augmentation để tăng cường dữ liệu.
Scheduler CosineAnnealingWarmRestarts cho learning rate.
Gradient clipping để kiểm soát gradient.
Lưu model tốt nhất dựa trên độ chính xác validation.
Vẽ đồ thị loss, accuracy, và confusion matrix.
Báo cáo chi tiết về quá trình huấn luyện và kiểm thử.

Lưu ý

Đảm bảo thư mục data/dataset/ có đủ dữ liệu trước khi huấn luyện.
Kiểm tra thiết bị hỗ trợ (cuda, mps, cpu) trước khi chạy.
Log và kết quả sẽ được lưu trong logs/ và results/.
