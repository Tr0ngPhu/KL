# data/combine_datasets.py
import os
import json
import shutil
import cv2
import re
import hashlib
from collections import defaultdict

def normalize_label(label):
    """Chuẩn hóa nhãn: bỏ dấu gạch ngang, dấu cách, và chuyển thành chữ thường"""
    return re.sub(r'[-_\s]+', '', label.lower())

def get_file_hash(filepath):
    """Tính MD5 hash của file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def convert_coco_to_classification(coco_json_path, image_dir, output_dir, class_map, use_bbox=False, seen_hashes=None):
    print(f"Processing {coco_json_path}")
    try:
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"JSON file not found: {coco_json_path}")
        return

    os.makedirs(os.path.join(output_dir, "Real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Fake"), exist_ok=True)

    normalized_class_map = {normalize_label(k): v for k, v in class_map.items()}
    print(f"Normalized class map: {normalized_class_map}")

    skipped_images = []
    processed_images = 0
    for image_info in coco_data["images"]:
        image_path = os.path.join(image_dir, image_info["file_name"])
        image_id = image_info["id"]

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Kiểm tra hash để tránh trùng lặp
        file_hash = get_file_hash(image_path)
        if file_hash in seen_hashes:
            print(f"Skipping duplicate image: {image_path} (hash: {file_hash})")
            continue
        seen_hashes.add(file_hash)

        target_class = None
        bbox = None
        for ann in coco_data["annotations"]:
            if ann["image_id"] == image_id:
                category_id = ann["category_id"]
                category_name = next(cat["name"] for cat in coco_data["categories"] if cat["id"] == category_id)
                normalized_category = normalize_label(category_name)
                target_class = normalized_class_map.get(normalized_category, None)
                if target_class:
                    bbox = ann["bbox"]
                    break

        if target_class is None:
            skipped_images.append((image_info["file_name"], category_name))
            continue

        output_path = os.path.join(output_dir, target_class, image_info["file_name"])
        try:
            if use_bbox and bbox:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Cannot read image: {image_path}")
                    continue
                x, y, w, h = bbox
                cropped_img = img[int(y):int(y+h), int(x):int(x+w)]
                cv2.imwrite(output_path, cropped_img)
                print(f"Cropped and copied {image_info['file_name']} to {target_class}")
            else:
                shutil.copy(image_path, output_path)
                print(f"Copied {image_info['file_name']} to {target_class}")
            processed_images += 1
        except Exception as e:
            print(f"Error processing {image_info['file_name']}: {str(e)}")

    print(f"Processed {processed_images} images")
    if skipped_images:
        with open(os.path.join(output_dir, "skipped_images.txt"), "a") as f:
            for img, cls in skipped_images:
                f.write(f"Skipped {img} due to unknown class: {cls}\n")
        print(f"Logged {len(skipped_images)} skipped images to {os.path.join(output_dir, 'skipped_images.txt')}")

# Ánh xạ nhãn cho từng dataset
class_maps = {
    "counterfeit-nike-shoes": {
        "nike original air force": "Real",
        "nike original jordan 1": "Real",
        "nike fake air force": "Fake",
        "nike fake jordan 1": "Fake",
        "shoes": None
    },
    "original-or-fake-shoes": {
        "ori air force": "Real",
        "ori jordan 1": "Real",
        "fake air force": "Fake",
        "fake jordan 1": "Fake",
        "shoes": None
    },
    "shoe-authentication-app": {
        "nike-original-air-force": "Real",
        "nike-original-jordan": "Real",
        "nike-fake-air-force": "Fake",
        "nike-fake-Jordan": "Fake",
        "shoes": None
    }
}

# Xóa thư mục dataset cũ để tránh trùng lặp tích lũy
if os.path.exists("data/dataset"):
    shutil.rmtree("data/dataset")
    print("Deleted old data/dataset directory")

# Khởi tạo tập hợp để theo dõi hash
seen_hashes = set()

# Chuyển đổi và hợp nhất
for split in ["train", "valid", "test"]:
    output_dir = f"data/dataset/{split if split != 'test' else 'test'}"
    print(f"\nProcessing split: {split}")
    for dataset in ["counterfeit-nike-shoes", "original-or-fake-shoes", "shoe-authentication-app"]:
        coco_json_path = f"data/{dataset}/{split}/_annotations.coco.json"
        image_dir = f"data/{dataset}/{split}"
        if os.path.exists(coco_json_path):
            convert_coco_to_classification(coco_json_path, image_dir, output_dir, class_maps[dataset], use_bbox=False, seen_hashes=seen_hashes)
        else:
            print(f"Skipping {dataset}/{split}: JSON file not found")