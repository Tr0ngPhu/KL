import os
import sys
import yaml
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torchvision.transforms as transforms

# Thêm đường dẫn gốc vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.model import VisionTransformer
from utils.utils import load_checkpoint, ensure_dir

# Initialize FastAPI app
app = FastAPI(title="Fake Product Detection API")

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(current_dir), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)

# Create upload directory
ensure_dir(UPLOAD_FOLDER)

# Load config
def load_config():
    config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load model
def load_model():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        path=os.path.join(os.path.dirname(current_dir), config['paths']['checkpoint_dir'], 'best_model.pt')
    )
    
    return model

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image for prediction
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Load model at startup
model = load_model()

@app.get("/")
async def root():
    return {"message": "Welcome to Product Authenticity Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        return {"error": "File format not supported. Please upload a JPG or PNG image."}
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][prediction].item()
        
        # Convert prediction to label
        predicted_label = 'Thật' if prediction.item() == 1 else 'Giả'
        
        return {
            "prediction": predicted_label,
            "confidence": confidence * 100,
            "probabilities": {
                "Giả": probabilities[0][0].item() * 100,
                "Thật": probabilities[0][1].item() * 100
            }
        }
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)