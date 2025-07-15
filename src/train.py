import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import logging
import yaml
from tqdm import tqdm
from datetime import datetime

# Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(os.path.dirname(current_dir), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

from models.model import VisionTransformer

class ProductDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        for class_name in ['fake', 'real']:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(1 if class_name == 'real' else 0)
        
        logging.info(f'Images: {len(self.images)} (Real: {sum(self.labels)}, Fake: {len(self.labels) - sum(self.labels)})')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            return None, None

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Bắt buộc dùng GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng! Cần GPU để train.")
        
        self.device = torch.device('cuda')
        self.model.to(self.device)
        
        # Mixed precision cho GPU
        self.scaler = GradScaler('cuda')
        self.accumulation_steps = 8
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=config['training']['learning_rate'] * 10,
            epochs=config['training']['epochs'],
            steps_per_epoch=len(train_loader) // self.accumulation_steps
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        
        # Results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(os.path.dirname(current_dir), 'results', timestamp)
        os.makedirs(self.results_dir, exist_ok=True)
        
        logging.info(f'Training on: {self.device}')
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        torch.cuda.empty_cache()
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch+1}')):
            if inputs is None: continue
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            with autocast('cuda'):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels) / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                if inputs is None: continue
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                with autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def save_checkpoint(self, epoch, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }
        
        torch.save(checkpoint, os.path.join(self.results_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(self.results_dir, 'best_model.pth'))
            logging.info(f'New best: {val_acc:.2f}%')
    
    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            logging.info(f'Epoch {epoch+1}/{epochs}: Train: {train_acc:.2f}%, Val: {val_acc:.2f}%')
            
            self.save_checkpoint(epoch, val_acc)
            
            # Early stopping
            if val_acc < self.best_val_acc - 5.0 and epoch > 10:
                logging.info('Early stopping')
                break

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = ProductDataset(
        data_dir=os.path.join(current_dir, 'data', 'train'),
        transform=train_transform
    )
    
    val_dataset = ProductDataset(
        data_dir=os.path.join(current_dir, 'data', 'validation'),
        transform=val_transform
    )
    
    # Data loaders - cross-platform num_workers
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=num_workers)
    
    # Model
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=2,
        dim=512,
        depth=8,
        heads=8,
        mlp_dim=2048
    )
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    logging.info('Starting training...')
    trainer.train(config['training']['epochs'])

if __name__ == '__main__':
    main()
