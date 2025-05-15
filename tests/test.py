import os
import sys
import torch
import time
import logging
import yaml
import argparse
import pandas as pd
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Thêm đường dẫn gốc vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

# Import các module của dự án
from src.model import VisionTransformer
from src.utils import load_checkpoint

# Thiết lập logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'testing.log')),
        logging.StreamHandler()
    ]
)

def load_config(config_path=None):
    """Tải cấu hình từ file YAML"""
    if config_path is None:
        config_path = os.path.join(project_root, 'config', 'config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Đã tải cấu hình từ {config_path}")
        return config
    except Exception as e:
        logging.error(f"Lỗi khi tải file cấu hình: {str(e)}")
        raise

def get_device(device_type=None):
    """Xác định thiết bị tính toán tối ưu"""
    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    logging.info(f"Đang sử dụng thiết bị: {device}")
    return device

def show_device_info(device):
    """Hiển thị thông tin chi tiết về thiết bị"""
    device_info = {
        'device_type': device.type,
        'device_index': device.index if device.index is not None else 'N/A'
    }
    
    if device.type == 'cuda':
        device_info.update({
            'device_name': torch.cuda.get_device_name(device.index or 0),
            'cuda_capability': torch.cuda.get_device_capability(device.index or 0),
            'total_memory': f"{torch.cuda.get_device_properties(device.index or 0).total_memory / (1024**3):.2f} GB",
            'cuda_version': torch.version.cuda
        })
    elif device.type == 'mps':
        device_info.update({
            'device_name': 'Apple Silicon GPU'
        })
    
    print("\n=== THÔNG TIN THIẾT BỊ ===")
    for key, value in device_info.items():
        print(f"{key.upper():<20}: {value}")
    
    return device_info

def load_image(image_path, target_size=(224, 224)):
    """Tải và tiền xử lý ảnh với xử lý lỗi mạnh mẽ"""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            return transform(img).unsqueeze(0)  # Thêm batch dimension
    except Exception as e:
        logging.error(f"Không thể tải ảnh {image_path}: {str(e)}")
        return None

def predict_single_image(model, image_path, device):
    """Dự đoán cho một ảnh duy nhất"""
    image_tensor = load_image(image_path)
    if image_tensor is None:
        return None
    
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(image_tensor.to(device))
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
            
            return {
                'prediction': 'real' if pred == 1 else 'fake',
                'confidence': confidence * 100,
                'probabilities': {
                    'fake': probs[0][0].item() * 100,
                    'real': probs[0][1].item() * 100
                }
            }
        except Exception as e:
            logging.error(f"Lỗi khi dự đoán ảnh {image_path}: {str(e)}")
            return None

def evaluate_model(model, test_dir, device, batch_size=32):
    """Đánh giá mô hình trên toàn bộ tập test"""
    results = []
    metrics = {
        'total': 0,
        'correct': 0,
        'class_stats': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'inference_time': 0
    }
    
    # Kiểm tra cấu trúc thư mục
    if not os.path.exists(test_dir):
        logging.error(f"Thư mục test không tồn tại: {test_dir}")
        return None
        
    label_dirs = {
        'real': os.path.join(test_dir, 'real'),
        'fake': os.path.join(test_dir, 'fake')
    }
    
    # Chuẩn bị dữ liệu
    all_images = []
    for label, dir_path in label_dirs.items():
        if not os.path.exists(dir_path):
            continue
            
        for img_name in os.listdir(dir_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append((os.path.join(dir_path, img_name), label))
    
    if not all_images:
        logging.error("Không tìm thấy ảnh nào trong thư mục test")
        return None
    
    # Xử lý theo batch để tối ưu hiệu năng
    num_batches = (len(all_images) + batch_size - 1) // batch_size
    
    with tqdm(total=len(all_images), desc="Đang đánh giá", unit="ảnh") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(all_images))
            batch_items = all_images[start_idx:end_idx]
            
            # Tải và xử lý batch ảnh
            batch_tensors = []
            valid_items = []
            
            for img_path, true_label in batch_items:
                tensor = load_image(img_path)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    valid_items.append((img_path, true_label))
            
            if not batch_tensors:
                continue
                
            batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
            
            # Dự đoán
            start_time = time.time()
            with torch.no_grad():
                outputs = model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                confidences = torch.max(probs, dim=1)[0].cpu().numpy()
            
            metrics['inference_time'] += time.time() - start_time
            
            # Xử lý kết quả
            for i, (img_path, true_label) in enumerate(valid_items):
                pred_label = 'real' if preds[i] == 1 else 'fake'
                is_correct = (pred_label == true_label)
                
                results.append({
                    'image': os.path.basename(img_path),
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidences[i] * 100,
                    'is_correct': is_correct
                })
                
                metrics['total'] += 1
                metrics['class_stats'][true_label]['total'] += 1
                
                if is_correct:
                    metrics['correct'] += 1
                    metrics['class_stats'][true_label]['correct'] += 1
                
                pbar.update(1)
    
    # Tính toán các chỉ số
    if metrics['total'] > 0:
        metrics['accuracy'] = metrics['correct'] / metrics['total'] * 100
        metrics['avg_inference_time'] = metrics['inference_time'] / metrics['total']
        
        # Tính balanced accuracy
        class_acc = []
        for label, stats in metrics['class_stats'].items():
            if stats['total'] > 0:
                class_acc.append(stats['correct'] / stats['total'])
        
        metrics['balanced_accuracy'] = sum(class_acc) / len(class_acc) * 100 if class_acc else 0
    
    return {
        'results': results,
        'metrics': metrics,
        'report': generate_classification_report(results)
    }

def generate_classification_report(results):
    """Tạo báo cáo phân loại chi tiết"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    report = {
        'confusion_matrix': pd.crosstab(
            df['true_label'], df['pred_label'], 
            rownames=['Thực tế'], colnames=['Dự đoán'],
            margins=True
        ),
        'class_stats': df.groupby('true_label').agg({
            'is_correct': ['count', 'mean'],
            'confidence': ['mean', 'std']
        })
    }
    
    return report

def run_benchmark(model, test_dir, device, num_runs=10, batch_sizes=[1, 8, 16, 32, 64]):
    """Đánh giá hiệu năng mô hình trên các batch size khác nhau"""
    benchmark_results = []
    
    # Tìm ảnh để benchmark
    sample_images = []
    for label in ['real', 'fake']:
        label_dir = os.path.join(test_dir, label)
        if not os.path.exists(label_dir):
            continue
            
        for img_name in os.listdir(label_dir)[:10]:  # Lấy tối đa 10 ảnh mỗi loại
            img_path = os.path.join(label_dir, img_name)
            tensor = load_image(img_path)
            if tensor is not None:
                sample_images.append(tensor)
    
    if not sample_images:
        logging.error("Không đủ ảnh để chạy benchmark")
        return None
    
    # Warm-up
    model.eval()
    with torch.no_grad():
        _ = model(torch.cat(sample_images[:2], dim=0).to(device))
    
    # Đo lường hiệu năng
    for batch_size in batch_sizes:
        if batch_size > len(sample_images):
            continue
            
        batch_tensor = torch.cat(sample_images[:batch_size], dim=0).to(device)
        
        # Đo thời gian
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(batch_tensor)
        avg_time = (time.time() - start_time) / num_runs
        
        benchmark_results.append({
            'batch_size': batch_size,
            'avg_inference_time': avg_time,
            'images_per_second': batch_size / avg_time,
            'device': str(device)
        })
    
    return benchmark_results

def save_results_to_csv(results, output_dir, filename_prefix):
    """Lưu kết quả ra file CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu kết quả đánh giá
    if 'results' in results and results['results']:
        df_results = pd.DataFrame(results['results'])
        results_path = os.path.join(output_dir, f"{filename_prefix}_results.csv")
        df_results.to_csv(results_path, index=False)
        logging.info(f"Đã lưu kết quả chi tiết vào {results_path}")
    
    # Lưu metrics
    if 'metrics' in results:
        df_metrics = pd.DataFrame([results['metrics']])
        metrics_path = os.path.join(output_dir, f"{filename_prefix}_metrics.csv")
        df_metrics.to_csv(metrics_path, index=False)
        logging.info(f"Đã lưu metrics vào {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description="Công cụ kiểm thử mô hình phân loại ảnh thật/giả")
    parser.add_argument('--model_path', required=True, help="Đường dẫn đến file model đã huấn luyện")
    parser.add_argument('--test_dir', default='data/test', help="Thư mục chứa dữ liệu test")
    parser.add_argument('--config', default='../config/config.yaml', help="Đường dẫn đến file cấu hình")
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto', help="Thiết bị sử dụng")
    parser.add_argument('--batch_size', type=int, default=32, help="Kích thước batch khi đánh giá")
    parser.add_argument('--output_dir', default='results', help="Thư mục lưu kết quả")
    parser.add_argument('--benchmark', action='store_true', help="Chạy đánh giá hiệu năng")
    args = parser.parse_args()
    
    # Khởi tạo thiết bị
    device = get_device(args.device if args.device != 'auto' else None)
    device_info = show_device_info(device)
    
    # Tải cấu hình
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Không thể khởi tạo do lỗi cấu hình: {str(e)}")
        return
    
    # Tải mô hình
    try:
        model = VisionTransformer(
            image_size=config['model']['image_size'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            dim=config['model']['dim'],
            depth=config['model']['depth'],
            heads=config['model']['heads'],
            mlp_dim=config['model']['mlp_dim']
        ).to(device)
        
        load_checkpoint(model, args.model_path)
        logging.info("Đã tải mô hình thành công")
    except Exception as e:
        logging.error(f"Lỗi khi tải mô hình: {str(e)}")
        return
    
    # Chạy benchmark hiệu năng nếu được yêu cầu
    if args.benchmark:
        logging.info("Bắt đầu đánh giá hiệu năng...")
        benchmark_results = run_benchmark(model, args.test_dir, device)
        
        if benchmark_results:
            print("\n=== KẾT QUẢ BENCHMARK ===")
            df_benchmark = pd.DataFrame(benchmark_results)
            print(df_benchmark.to_markdown(tablefmt="grid"))
            
            # Lưu kết quả benchmark
            benchmark_path = os.path.join(args.output_dir, "benchmark_results.csv")
            df_benchmark.to_csv(benchmark_path, index=False)
            logging.info(f"Đã lưu kết quả benchmark vào {benchmark_path}")
    
    # Đánh giá mô hình trên tập test
    logging.info("Bắt đầu đánh giá mô hình...")
    eval_results = evaluate_model(model, args.test_dir, device, args.batch_size)
    
    if eval_results:
        # Hiển thị kết quả
        print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
        print(f"Tổng số ảnh: {eval_results['metrics']['total']}")
        print(f"Độ chính xác: {eval_results['metrics']['accuracy']:.2f}%")
        print(f"Độ chính xác cân bằng: {eval_results['metrics']['balanced_accuracy']:.2f}%")
        print(f"Thời gian suy luận trung bình: {eval_results['metrics']['avg_inference_time']:.4f}s/ảnh")
        
        # Hiển thị thống kê từng lớp
        print("\nTHỐNG KÊ THEO LỚP:")
        for label, stats in eval_results['metrics']['class_stats'].items():
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"{label.upper():<10}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
        
        # Hiển thị confusion matrix
        if eval_results['report']:
            print("\nMA TRẬN NHẦM LẪN:")
            print(eval_results['report']['confusion_matrix'].to_markdown(tablefmt="grid"))
        
        # Lưu kết quả
        save_results_to_csv(eval_results, args.output_dir, "evaluation")
        
        # Lưu các ảnh dự đoán sai
        wrong_predictions = [r for r in eval_results['results'] if not r['is_correct']]
        if wrong_predictions:
            wrong_path = os.path.join(args.output_dir, "wrong_predictions.csv")
            pd.DataFrame(wrong_predictions).to_csv(wrong_path, index=False)
            logging.info(f"Đã lưu danh sách ảnh dự đoán sai vào {wrong_path}")

if __name__ == '__main__':
    main()