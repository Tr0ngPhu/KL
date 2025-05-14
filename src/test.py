import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import argparse
import logging
import yaml
from torch.utils.data import DataLoader

# Thêm đường dẫn gốc vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from data.dataset import CustomDataset
from models.model import VisionTransformer
from utils.utils import ensure_dir, setup_logging, load_checkpoint

from models.model import load_model, predict_image

# Thiết lập logging
os.makedirs('../logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('../logs', 'testing.log')),
        logging.StreamHandler()
    ]
)

def load_config():
    config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_image(image_path):
    """Load và tiền xử lý ảnh"""
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)  # Thêm batch dimension
        return image_tensor
    except Exception as e:
        logging.error(f"Lỗi khi đọc ảnh {image_path}: {str(e)}")
        return None

def predict_single_image(model, image_path, device='cuda'):
    """Dự đoán một ảnh"""
    # Load và tiền xử lý ảnh
    image_tensor = load_image(image_path)
    if image_tensor is None:
        return None
    
    # Chuyển ảnh lên device
    image_tensor = image_tensor.to(device)
    
    # Dự đoán
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0][prediction].item()
        
    # Xác định nhãn thật từ đường dẫn
    true_label = 'Thật' if 'real' in image_path else 'Giả'
    predicted_label = 'Thật' if prediction.item() == 1 else 'Giả'
    is_correct = true_label == predicted_label
        
    # Chuyển đổi kết quả
    result = {
        'true_label': true_label,
        'prediction': predicted_label,
        'is_correct': is_correct,
        'confidence': confidence * 100,
        'probabilities': {
            'Giả': probabilities[0][0].item() * 100,
            'Thật': probabilities[0][1].item() * 100
        }
    }
    
    return result

def test_directory(model, test_dir, device='cuda'):
    """Kiểm tra toàn bộ thư mục test"""
    results = []
    total = 0
    correct = 0
    
    # Duyệt qua các thư mục con (real/fake)
    for label in ['real', 'fake']:
        label_dir = os.path.join(test_dir, label)
        if not os.path.exists(label_dir):
            continue
            
        # Duyệt qua các ảnh trong thư mục
        for img_name in os.listdir(label_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(label_dir, img_name)
            result = predict_single_image(model, img_path, device)
            
            if result is None:
                continue
                
            # Kiểm tra kết quả
            true_label = 'Thật' if label == 'real' else 'Giả'
            is_correct = result['prediction'] == true_label
            
            results.append({
                'image': img_name,
                'true_label': true_label,
                'predicted_label': result['prediction'],
                'confidence': result['confidence'],
                'is_correct': is_correct
            })
            
            total += 1
            if is_correct:
                correct += 1
    
    # Tính toán độ chính xác
    accuracy = (correct / total * 100) if total > 0 else 0
    
    return {
        'results': results,
        'total': total,
        'correct': correct,
        'accuracy': accuracy
    }

def main():
    parser = argparse.ArgumentParser(description='Kiểm tra mô hình phân loại sản phẩm thật/giả')
    parser.add_argument('--model_path', type=str, required=True, help='Đường dẫn đến file model đã huấn luyện')
    parser.add_argument('--test_dir', type=str, default=os.path.join(current_dir, 'data', 'raw', 'test'), help='Thư mục chứa ảnh test')
    parser.add_argument('--single_image', type=str, help='Đường dẫn đến ảnh cần kiểm tra')
    args = parser.parse_args()
    
    # Kiểm tra GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Sử dụng device: {device}')
    
    # Load config
    config = load_config()
    
    # Load model
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
        
        checkpoint = load_checkpoint(
            model=model,
            path=args.model_path
        )
        logging.info('Đã load model thành công')
    except Exception as e:
        logging.error(f'Lỗi khi load model: {str(e)}')
        return
    
    # Kiểm tra một ảnh đơn lẻ
    if args.single_image:
        if not os.path.exists(args.single_image):
            logging.error(f'Không tìm thấy ảnh: {args.single_image}')
            return
            
        result = predict_single_image(model, args.single_image, device)
        if result:
            print('\nKết quả dự đoán:')
            print(f'Ảnh: {os.path.basename(args.single_image)}')
            print(f'Ảnh này là hàng thật/giả: {result["true_label"]}')
            print(f'Dự đoán: {"Đúng" if result["is_correct"] else "Sai"}')
            print(f'Độ tin cậy: {result["confidence"]:.2f}%')
            print('\nXác suất chi tiết:')
            print(f'Giả: {result["probabilities"]["Giả"]:.2f}%')
            print(f'Thật: {result["probabilities"]["Thật"]:.2f}%')
    
    # Kiểm tra toàn bộ thư mục test
    if os.path.exists(args.test_dir):
        print('\nKiểm tra toàn bộ thư mục test:')
        test_results = test_directory(model, args.test_dir, device)
        
        print(f'\nTổng số ảnh: {test_results["total"]}')
        print(f'Số ảnh đúng: {test_results["correct"]}')
        print(f'Độ chính xác: {test_results["accuracy"]:.2f}%')
        
        # In chi tiết các ảnh bị dự đoán sai
        print('\nChi tiết các ảnh bị dự đoán sai:')
        for result in test_results['results']:
            if not result['is_correct']:
                print(f"\nẢnh: {result['image']}")
                print(f'Ảnh này là hàng thật/giả: {result["true_label"]}')
                print(f'Dự đoán: Sai')
                print(f'Độ tin cậy: {result["confidence"]:.2f}%')

if __name__ == '__main__':
    main() 