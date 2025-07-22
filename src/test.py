import os
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchvision import transforms
from PIL import Image
import yaml
import logging
from datetime import datetime

# Thiết lập
current_dir = os.path.dirname(os.path.abspath(__file__))
from models.model import VisionTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Biến toàn cục
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
transform = None
class_names = ['Giả', 'Thật']

def load_config():
    """Load cấu hình từ file config.yaml"""
    config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_transforms():
    """Thiết lập transforms cho test"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def find_latest_checkpoint():
    """Tìm checkpoint mới nhất"""
    results_dir = os.path.join(os.path.dirname(current_dir), 'results')
    
    # Sử dụng checkpoint mới nhất với 90.6% accuracy
    latest_checkpoint = os.path.join(results_dir, '20250720_053233', 'best_model.pth')
    if os.path.exists(latest_checkpoint):
        print(f"✅ Found best checkpoint: 90.6% accuracy")
        return latest_checkpoint
    
    # Fallback: tìm checkpoint mới nhất khác
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
            logging.error('Không tìm thấy checkpoint để test!')
            return False
        
        model.to(device)
        model.eval()
        transform = setup_transforms()
        
        logging.info('Model đã load thành công!')
        return True
        
    except Exception as e:
        logging.error(f'Lỗi khi load model: {e}')
        return False

def test_single_image(image_path):
    """Test một ảnh đơn lẻ"""
    try:
        # Load và preprocess ảnh
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Dự đoán
        with torch.no_grad():
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = predicted.cpu().item()
        confidence_score = confidence.cpu().item()
        
        result = {
            'image_path': image_path,
            'is_real': bool(predicted_class == 1),
            'class': class_names[predicted_class],
            'confidence': round(confidence_score * 100, 2),
            'probabilities': {
                'fake': round(probabilities[0][0].cpu().item() * 100, 2),
                'real': round(probabilities[0][1].cpu().item() * 100, 2)
            }
        }
        
        return result
        
    except Exception as e:
        logging.error(f'Lỗi khi test ảnh {image_path}: {e}')
        return None

def test_folder(test_folder_path):
    """Test tất cả ảnh trong một folder"""
    if not os.path.exists(test_folder_path):
        logging.error(f'Folder không tồn tại: {test_folder_path}')
        return
    
    results = []
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    for filename in os.listdir(test_folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(test_folder_path, filename)
            result = test_single_image(image_path)
            if result:
                results.append(result)
                logging.info(f'{filename}: {result["class"]} ({result["confidence"]}%)')
    
    # Thống kê
    if results:
        total = len(results)
        real_count = sum(1 for r in results if r['is_real'])
        fake_count = total - real_count
        
        logging.info(f'\n=== THỐNG KÊ TEST ===')
        logging.info(f'Tổng số ảnh: {total}')
        logging.info(f'Dự đoán THẬT: {real_count} ({real_count/total*100:.1f}%)')
        logging.info(f'Dự đoán GIẢ: {fake_count} ({fake_count/total*100:.1f}%)')
        logging.info(f'Confidence trung bình: {sum(r["confidence"] for r in results)/total:.2f}%')
    
    return results

def main():
    """Hàm chính để test"""
    logging.info('=== KHỞI ĐỘNG TEST ===')
    
    # Load model
    if not load_model():
        logging.error('Không thể load model để test!')
        return
    
    # Test một ảnh đơn lẻ (thay đổi đường dẫn)
    # single_image_path = "path/to/your/test/image.jpg"
    # result = test_single_image(single_image_path)
    # if result:
    #     print(f"Kết quả: {result}")
    
    # Test folder chứa ảnh test
    test_folder_path = os.path.join(current_dir, 'data', 'test', 'fake')
    if os.path.exists(test_folder_path):
        logging.info(f'Testing folder FAKE: {test_folder_path}')
        test_folder(test_folder_path)
    
    test_folder_path = os.path.join(current_dir, 'data', 'test', 'real')
    if os.path.exists(test_folder_path):
        logging.info(f'Testing folder REAL: {test_folder_path}')
        test_folder(test_folder_path)
    
    logging.info('=== TEST HOÀN THÀNH ===')

if __name__ == '__main__':
    main()
