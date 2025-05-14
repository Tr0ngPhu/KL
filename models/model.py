import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_B_16_Weights

class ProductAuthenticityModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super(ProductAuthenticityModel, self).__init__()
        
        # Sử dụng Vision Transformer (ViT) làm backbone
        self.vit = models.vit_b_16(
            weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Lấy số features từ backbone
        num_features = self.vit.heads.head.in_features
        
        # Thêm các lớp mới với dropout và batch normalization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),  # Sử dụng GELU thay vì ReLU cho ViT
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Thay thế classifier cũ
        self.vit.heads = nn.Sequential(
            nn.LayerNorm(num_features),
            self.classifier
        )
        
    def forward(self, x):
        return self.vit(x)

def load_model(model_path):
    """Load model với các tham số đã lưu"""
    model = ProductAuthenticityModel()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_image(model, image_tensor):
    """Dự đoán một ảnh và trả về kết quả"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        return prediction.item(), probabilities[0].tolist()
