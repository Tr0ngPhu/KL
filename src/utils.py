import os
import logging
import torch

def ensure_dir(directory):
    """Tạo thư mục nếu chưa tồn tại"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_logging(log_file):
    """Thiết lập logging"""
    ensure_dir(os.path.dirname(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def check_admin_access():
    """Kiểm tra quyền admin"""
    try:
        return os.getuid() == 0
    except AttributeError:
        return False

def save_checkpoint(model, optimizer, epoch, path):
    """Lưu checkpoint của model"""
    ensure_dir(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, path):
    """Load checkpoint của model"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model 