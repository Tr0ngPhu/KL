import subprocess
import sys
import os
from pathlib import Path
import platform

# CẤU HÌNH CHÍNH

REQUIRED_PYTHON_VERSION = (3, 6)
BASE_DIR = Path(__file__).parent

FOLDER_STRUCTURE = [
    'data/train/real', 'data/train/fake',
    'data/validation/real', 'data/validation/fake',
    'data/test/real', 'data/test/fake',
    'models', 'results', 'static', 'templates',
    'logs', 'config', 'utils'
]

REQUIREMENTS = {
    'data_processing': [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'opencv-python>=4.5.0',
        'pillow>=9.0.0'
    ],
    'machine_learning': [
        'tensorflow>=2.8.0',
        'scikit-learn>=1.0.0',
        'keras>=2.8.0'
    ],
    'visualization': [
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0'
    ],
    'utilities': [
        'tqdm>=4.62.0',
        'python-dotenv>=0.19.0',
        'flask>=2.0.0'
    ]
}


# CÁC HÀM TIỆN ÍCH


def check_python_version():
    """Kiểm tra phiên bản Python"""
    if sys.version_info < REQUIRED_PYTHON_VERSION:
        sys.exit(f"Yêu cầu Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]} hoặc cao hơn")

def create_folders():
    """Tạo cấu trúc thư mục"""
    print("\n" + "="*40)
    print("Tạo cấu trúc thư mục")
    print("="*40)
    
    for folder in FOLDER_STRUCTURE:
        path = BASE_DIR / folder
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Tạo thành công: {folder}")
            (path / '.gitkeep').touch(exist_ok=True)  # Thêm file .gitkeep cho thư mục rỗng
        except OSError as e:
            print(f"✗ Lỗi khi tạo {folder}: {str(e)}")

def install_packages():
    """Cài đặt các gói Python cần thiết"""
    print("\n" + "="*40)
    print("Cài đặt các gói Python")
    print("="*40)
    
    # Cập nhật pip trước
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    except subprocess.CalledProcessError:
        print("⚠ Cảnh báo: Không thể cập nhật pip")

    # Cài đặt tất cả requirements
    all_packages = [pkg for category in REQUIREMENTS.values() for pkg in category]
    
    for package in all_packages:
        try:
            print(f"\n▶ Đang cài đặt {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Thành công: {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Thất bại: Không thể cài đặt {package}")

def generate_requirements():
    """Tạo file requirements.txt"""
    print("\n" + "="*40)
    print("Tạo file requirements.txt")
    print("="*40)
    
    with open(BASE_DIR / 'requirements.txt', 'w') as f:
        for category, packages in REQUIREMENTS.items():
            f.write(f"\n# {category.upper()}\n")
            f.write('\n'.join(packages))
    
    print("✓ Đã tạo file requirements.txt")

# ================================================
# MAIN SCRIPT
# ================================================

def setup_environment():
    """Hàm chính thiết lập môi trường"""
    print("🛠 Thiết lập môi trường cho dự án phát hiện hàng giả")
    
    check_python_version()
    create_folders()
    install_packages()
    generate_requirements()
    
    print("\n✅ Thiết lập hoàn tất!")

if __name__ == "__main__":
    setup_environment()