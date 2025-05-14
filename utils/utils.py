import os
import sys
import ctypes
import logging
from pathlib import Path
import torch

def is_admin():
    """Kiểm tra xem chương trình có đang chạy với quyền admin không"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    """Khởi động lại chương trình với quyền admin"""
    try:
        if sys.argv[-1] != 'asadmin':
            script = os.path.abspath(sys.argv[0])
            params = ' '.join([script] + sys.argv[1:] + ['asadmin'])
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, None, 1)
            sys.exit()
    except Exception as e:
        logging.error(f"Lỗi khi yêu cầu quyền admin: {str(e)}")
        return False
    return True

def check_admin_access(path):
    """Kiểm tra và yêu cầu quyền admin nếu cần thiết để truy cập path"""
    try:
        # Thử tạo một file test để kiểm tra quyền truy cập
        test_file = Path(path) / '.test_write_access'
        test_file.touch()
        test_file.unlink()
        return True
    except PermissionError:
        if not is_admin():
            logging.warning(f"Cần quyền admin để truy cập {path}")
            return run_as_admin()
        return False
    except Exception as e:
        logging.error(f"Lỗi khi kiểm tra quyền truy cập {path}: {str(e)}")
        return False

def ensure_dir(path):
    """Tạo thư mục nếu chưa tồn tại, yêu cầu quyền admin nếu cần"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except PermissionError:
        if not is_admin():
            logging.warning(f"Cần quyền admin để tạo thư mục {path}")
            return run_as_admin()
        return False
    except Exception as e:
        logging.error(f"Lỗi khi tạo thư mục {path}: {str(e)}")
        return False

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
    return checkpoint 