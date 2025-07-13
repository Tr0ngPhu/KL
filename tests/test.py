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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import psutil
import numpy as np
# Thêm đường dẫn gốc vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

# Import các module của dự án
from src.explainable_model import ExplainableVisionTransformer, ExplainabilityAnalyzer  # Sử dụng đúng mô hình
from src.utils import ensure_dir

# Thiết lập logging
log_dir = os.path.join(project_root, 'logs')
ensure_dir(log_dir)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'testing.log')),
        logging.StreamHandler()
    ]
)

def validate_config(config):
    """Kiểm tra cấu hình YAML có đầy đủ và hợp lệ"""
    required_keys = [
        ('model.image_size', int),
        ('model.patch_size', int),
        ('model.num_classes', int),
        ('model.dim', int),
        ('model.depth', int),
        ('model.heads', int),
        ('model.mlp_dim', int)
    ]
    for key, expected_type in required_keys:
        keys = key.split('.')
        value = config
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                raise KeyError(f"Missing or invalid config key: {key}")
            value = value[k]
        try:
            value = expected_type(value)
        except (ValueError, TypeError):
            raise ValueError(f"Config key {key} must be convertible to {expected_type.__name__}, got {value}")
    logging.info("Cấu hình hợp lệ")

def load_config(config_path=None):
    """Tải cấu hình từ file YAML"""
    if config_path is None:
        config_path = os.path.join(project_root, 'config', 'config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        validate_config(config)
        logging.info(f"Đã tải cấu hình từ {config_path}")
        return config
    except Exception as e:
        logging.error(f"Lỗi khi tải file cấu hình: {str(e)}")
        raise

def get_device(device_type=None):
    """Xác định thiết bị tính toán tối ưu"""
    if device_type == 'cuda' and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_type == 'mps' and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_type == 'cpu':
        return torch.device("cpu")
    else:
        logging.warning(f"Thiết bị {device_type} không khả dụng, chọn tự động")
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

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
    
    mem = psutil.virtual_memory()
    device_info.update({
        'cpu_usage': f"{psutil.cpu_percent()}%",
        'memory_used': f"{mem.used / (1024**3):.2f} GB",
        'memory_available': f"{mem.available / (1024**3):.2f} GB"
    })
    
    print("\n=== THÔNG TIN THIẾT BỊ ===")
    for key, value in device_info.items():
        print(f"{key.upper():<20}: {value}")
    
    return device_info

def load_checkpoint(model, checkpoint_path, device):
    """Tải checkpoint mô hình"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict'")
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Đã tải checkpoint từ {checkpoint_path}")
    except Exception as e:
        logging.error(f"Lỗi khi tải checkpoint: {str(e)}")
        raise
def load_image(image_path, target_size=(224, 224)):
    """Tải và tiền xử lý ảnh"""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = transform(img)  # Shape: (C, H, W)
            original = np.array(img.resize(target_size))
            logging.info(f"Loaded image: {image_path}, tensor shape: {tensor.shape}, original shape: {original.shape}")
            return tensor, original
    except Exception as e:
        logging.error(f"Không thể tải ảnh {image_path}: {str(e)}")
        return None, None

def predict_single_image(model, image_path, device, analyzer):
    """Dự đoán cho một ảnh duy nhất với giải thích"""
    image_tensor, original_image = load_image(image_path)
    if image_tensor is None or original_image is None:
        return None
    
    model.eval()
    result = analyzer.predict_with_explanation(image_tensor.to(device), original_image)
    
    save_path = os.path.join('results', 'predictions', f"prediction_{os.path.basename(image_path)}.png")
    analyzer.visualize_explanation(image_tensor, original_image, result, save_path)
    
    return result


def evaluate_model(model, test_dir, device, batch_size=32, label_csv=None, num_workers=4):
    """Đánh giá mô hình trên toàn bộ tập test"""
    analyzer = ExplainabilityAnalyzer(model, class_names=['Real', 'Fake'])
    results = []
    metrics = {
        'total': 0,
        'correct': 0,
        'class_stats': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'inference_time': 0,
        'skipped_images': 0
    }
    
    all_images = []
    if label_csv:
        try:
            df = pd.read_csv(label_csv)
            for _, row in df.iterrows():
                img_path = os.path.join(test_dir, row['image'])
                if os.path.exists(img_path):
                    all_images.append((img_path, row['label'].lower()))
                else:
                    logging.warning(f"Ảnh không tồn tại: {img_path}")
                    metrics['skipped_images'] += 1
        except Exception as e:
            logging.error(f"Lỗi khi đọc file CSV: {str(e)}")
            return None
    else:
        label_dirs = {
            'real': os.path.join(test_dir, 'Real'),
            'fake': os.path.join(test_dir, 'Fake')
        }
        for label, dir_path in label_dirs.items():
            if not os.path.exists(dir_path):
                logging.warning(f"Thư mục không tồn tại: {dir_path}")
                continue
            for img_name in os.listdir(dir_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append((os.path.join(dir_path, img_name), label))
    
    if not all_images:
        logging.error("Không tìm thấy ảnh nào trong thư mục test")
        return None
    
    class_counts = {'real': 0, 'fake': 0}
    for _, label in all_images:
        class_counts[label] += 1
    logging.info(f"Phân phối lớp: Real={class_counts['real']}, Fake={class_counts['fake']}")
    
    num_batches = (len(all_images) + batch_size - 1) // batch_size
    
    with tqdm(total=len(all_images), desc="Đang đánh giá", unit="ảnh") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(all_images))
            batch_items = all_images[start_idx:end_idx]
            
            batch_tensors = []
            batch_originals = []
            valid_items = []
            
            for img_path, true_label in batch_items:
                tensor, original = load_image(img_path)
                if tensor is not None and original is not None:
                    batch_tensors.append(tensor)
                    batch_originals.append(original)
                    valid_items.append((img_path, true_label))
                else:
                    metrics['skipped_images'] += 1
                    pbar.update(1)
                    continue
            
            if not batch_tensors:
                continue
                
            batch_tensor = torch.stack(batch_tensors).to(device)  # Shape: (batch_size, C, H, W)
            logging.info(f"Batch tensor shape: {batch_tensor.shape}")
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                confidences = torch.max(probs, dim=1)[0].cpu().numpy()
            
            metrics['inference_time'] += time.time() - start_time
            
            for i, (img_path, true_label) in enumerate(valid_items):
                pred_label = 'real' if preds[i] == 1 else 'fake'
                is_correct = (pred_label == true_label)
                
                # Truyền trực tiếp batch_tensors[i] mà không cần unsqueeze
                result = analyzer.predict_with_explanation(batch_tensors[i].to(device), batch_originals[i])
                save_path = os.path.join('results', 'predictions', f"prediction_{os.path.basename(img_path)}.png")
                analyzer.visualize_explanation(batch_tensors[i], batch_originals[i], result, save_path)
                
                results.append({
                    'image': os.path.basename(img_path),
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidences[i] * 100,
                    'is_correct': is_correct,
                    'explanation': result['explanation']
                })
                
                metrics['total'] += 1
                metrics['class_stats'][true_label]['total'] += 1
                
                if is_correct:
                    metrics['correct'] += 1
                    metrics['class_stats'][true_label]['correct'] += 1
                
                pbar.update(1)
    
    if metrics['total'] > 0:
        metrics['accuracy'] = metrics['correct'] / metrics['total'] * 100
        metrics['avg_inference_time'] = metrics['inference_time'] / metrics['total']
        
        class_acc = []
        for label, stats in metrics['class_stats'].items():
            if stats['total'] > 0:
                class_acc.append(stats['correct'] / stats['total'])
        metrics['balanced_accuracy'] = sum(class_acc) / len(class_acc) * 100 if class_acc else 0
    
    logging.info(f"Số ảnh bị bỏ qua do lỗi: {metrics['skipped_images']}")
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
    y_true = df['true_label'].map({'real': 1, 'fake': 0})
    y_pred = df['pred_label'].map({'real': 1, 'fake': 0})
    
    report = {
        'confusion_matrix': pd.crosstab(
            df['true_label'], df['pred_label'],
            rownames=['Thực tế'], colnames=['Dự đoán'],
            margins=True
        ),
        'class_stats': df.groupby('true_label').agg({
            'is_correct': ['count', 'mean'],
            'confidence': ['mean', 'std']
        }),
        'classification_report': classification_report(
            y_true, y_pred, target_names=['fake', 'real'], output_dict=True
        )
    }
    
    return report

def plot_confusion_matrix(report, output_dir):
    """Vẽ confusion matrix"""
    cm = report['confusion_matrix'].iloc[:-1, :-1]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['fake', 'real'],
                yticklabels=['fake', 'real'])
    plt.title('Confusion Matrix')
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def run_benchmark(model, test_dir, device, num_runs=10, batch_sizes=[1, 8, 16, 32]):
    """Đánh giá hiệu năng mô hình"""
    benchmark_results = []
    
    sample_images = []
    for label in ['real', 'fake']:
        label_dir = os.path.join(test_dir, label)
        if not os.path.exists(label_dir):
            continue
        for img_name in os.listdir(label_dir)[:100]:
            img_path = os.path.join(label_dir, img_name)
            tensor, _ = load_image(img_path)
            if tensor is not None:
                sample_images.append(tensor)
    
    if not sample_images:
        logging.error("Không đủ ảnh để chạy benchmark")
        return None
    
    model.eval()
    with torch.no_grad():
        _ = model(torch.cat(sample_images[:2], dim=0).to(device))
    
    for batch_size in batch_sizes:
        if batch_size > len(sample_images):
            continue
            
        batch_tensor = torch.cat(sample_images[:batch_size], dim=0).to(device)
        
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
    ensure_dir(output_dir)
    
    if 'results' in results and results['results']:
        df_results = pd.DataFrame(results['results'])
        results_path = os.path.join(output_dir, f"{filename_prefix}_results.csv")
        df_results.to_csv(results_path, index=False)
        logging.info(f"Đã lưu kết quả chi tiết vào {results_path}")
    
    if 'metrics' in results:
        df_metrics = pd.DataFrame([results['metrics']])
        metrics_path = os.path.join(output_dir, f"{filename_prefix}_metrics.csv")
        df_metrics.to_csv(metrics_path, index=False)
        logging.info(f"Đã lưu metrics vào {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description="Công cụ kiểm thử mô hình phân loại ảnh thật/giả")
    parser.add_argument('--model_path', required=True, help="Đường dẫn đến file model đã huấn luyện")
    parser.add_argument('--test_dir', default='../data/dataset/test', help="Thư mục chứa dữ liệu test")
    parser.add_argument('--label_csv', default=None, help="File CSV chứa nhãn test (cột: image, label)")
    parser.add_argument('--image_path', default=None, help="Đường dẫn đến ảnh cần dự đoán")
    parser.add_argument('--config', default='../config/config.yaml', help="Đường dẫn đến file cấu hình")
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto', help="Thiết bị sử dụng")
    parser.add_argument('--batch_size', type=int, default=32, help="Kích thước batch khi đánh giá")
    parser.add_argument('--num_workers', type=int, default=2, help="Số worker cho DataLoader")
    parser.add_argument('--output_dir', default='results', help="Thư mục lưu kết quả")
    parser.add_argument('--benchmark', action='store_true', help="Chạy đánh giá hiệu năng")
    args = parser.parse_args()
    
    device = get_device(args.device if args.device != 'auto' else None)
    show_device_info(device)
    
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Không thể khởi tạo do lỗi cấu hình: {str(e)}")
        return
    
    try:
        model = ExplainableVisionTransformer(
            image_size=config['model']['image_size'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            dim=config['model']['dim'],
            depth=config['model']['depth'],
            heads=config['model']['heads'],
            mlp_dim=config['model']['mlp_dim'],
            dropout=config['training'].get('dropout', 0.2)
        ).to(device)
        load_checkpoint(model, args.model_path, device)
    except Exception as e:
        logging.error(f"Lỗi khi tải mô hình: {str(e)}")
        return
    
    if args.image_path:
        analyzer = ExplainabilityAnalyzer(model, class_names=['Real', 'Fake'])
        result = predict_single_image(model, args.image_path, device, analyzer)
        if result:
            print("\n=== KẾT QUẢ DỰ ĐOÁN ===")
            print(f"Dự đoán: {result['prediction']}")
            print(f"Độ tin cậy: {result['confidence']:.2f}%")
            print(f"Xác suất: Fake={result['probabilities'][0]:.2f}%, Real={result['probabilities'][1]:.2f}%")
            print("\n=== GIẢI THÍCH ===")
            print(result['explanation'])
        return
    
    if args.benchmark:
        logging.info("Bắt đầu đánh giá hiệu năng...")
        benchmark_results = run_benchmark(model, args.test_dir, device)
        if benchmark_results:
            print("\n=== KẾT QUẢ BENCHMARK ===")
            df_benchmark = pd.DataFrame(benchmark_results)
            print(df_benchmark.to_markdown(tablefmt="grid"))
            benchmark_path = os.path.join(args.output_dir, "benchmark_results.csv")
            df_benchmark.to_csv(benchmark_path, index=False)
            logging.info(f"Đã lưu kết quả benchmark vào {benchmark_path}")
    
    logging.info("Bắt đầu đánh giá mô hình...")
    eval_results = evaluate_model(model, args.test_dir, device, args.batch_size, args.label_csv, args.num_workers)
    
    if eval_results:
        print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
        print(f"Tổng số ảnh: {eval_results['metrics']['total']}")
        print(f"Độ chính xác: {eval_results['metrics']['accuracy']:.2f}%")
        print(f"Độ chính xác cân bằng: {eval_results['metrics']['balanced_accuracy']:.2f}%")
        print(f"Thời gian suy luận trung bình: {eval_results['metrics']['avg_inference_time']:.4f}s/ảnh")
        
        print("\nTHỐNG KÊ THEO LỚP:")
        for label, stats in eval_results['metrics']['class_stats'].items():
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"{label.upper():<10}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
        
        if eval_results['report']:
            print("\nMA TRẬN NHẦM LẪN:")
            print(eval_results['report']['confusion_matrix'].to_markdown(tablefmt="grid"))
            print("\nBÁO CÁO PHÂN LOẠI:")
            print(pd.DataFrame(eval_results['report']['classification_report']).T.to_markdown(tablefmt="grid"))
            plot_confusion_matrix(eval_results['report'], args.output_dir)
        
        save_results_to_csv(eval_results, args.output_dir, "evaluation")
        
        wrong_predictions = [r for r in eval_results['results'] if not r['is_correct']]
        if wrong_predictions:
            wrong_path = os.path.join(args.output_dir, "wrong_predictions.csv")
            pd.DataFrame(wrong_predictions).to_csv(wrong_path, index=False)
            logging.info(f"Đã lưu danh sách ảnh dự đoán sai vào {wrong_path}")

if __name__ == '__main__':
    main()