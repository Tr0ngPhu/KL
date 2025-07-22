"""
Clean API for KL Fake Detection with Smart Explainer
Optimized and simple
"""

import os
import io
from datetime import datetime
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
from explainer import ExplainabilityAnalyzer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = FastAPI(title="KL Fake Detection API", version="5.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
analyzer = None
transform = None
class_names = ['Fake', 'Real']

def load_best_model():
    """Load the best trained model from the latest results folder."""
    global model, analyzer
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(os.path.dirname(current_dir), 'results')
        
        if not os.path.exists(results_dir):
            print("⚠️ Results directory not found.")
            return False

        # Find the most recent training folder by sorting alphabetically/numerically
        all_folders = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        if not all_folders:
            print("⚠️ No training results found.")
            return False
            
        latest_folder = max(all_folders)
        model_path = os.path.join(results_dir, latest_folder, 'best_model.pth')

        if os.path.exists(model_path):
            # Load model architecture from config to ensure consistency
            config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            model_name = config['model'].get('name', 'vit_base_patch16_224')
            num_classes = config['model']['num_classes']

            # Create model instance without pretrained weights, as we'll load our own
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
            
            # Load the state dictionary from our checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Initialize the correct analyzer with the loaded model
            analyzer = ExplainabilityAnalyzer(model, class_names)
            accuracy = checkpoint.get('accuracy', 'N/A')
            if isinstance(accuracy, float):
                print(f"✅ Loaded best model from '{latest_folder}' with accuracy {accuracy:.2%}")
            else:
                print(f"✅ Loaded best model from '{latest_folder}' (accuracy not recorded in checkpoint).")
            return True
        else:
            print(f"⚠️ 'best_model.pth' not found in the latest folder '{latest_folder}'.")
            return False

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def fallback_to_pretrained():
    """If loading local model fails, use a pretrained one as a fallback."""
    global model, analyzer
    print("⚠️ Falling back to a generic pretrained model.")
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
    model.to(device)
    model.eval()
    analyzer = ExplainabilityAnalyzer(model, class_names)
    print("✅ Loaded pretrained 'vit_base_patch16_224'.")

def setup_transform():
    """Setup image transforms"""
    global transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# --- Initialization ---
setup_transform()
if not load_best_model():
    fallback_to_pretrained()

# Serve static files
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(os.path.dirname(current_dir), "web")
uploads_dir = os.path.join(os.path.dirname(current_dir), "uploads")

if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    try:
        index_path = os.path.join(web_dir, "index.html")
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading UI: {str(e)}</h1>", status_code=500)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict and explain an image"""
    if not model or not analyzer:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Transform - ensure correct tensor shape [1, 3, 224, 224]
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Get prediction and explanation
        result = analyzer.predict_with_explanation(img_tensor, np.array(image))

        # Save visualizations
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        heatmap_path = os.path.join(uploads_dir, f'heatmap_{timestamp}.png')
        analysis_path = os.path.join(uploads_dir, f'analysis_{timestamp}.png')
        
        # Save heatmap
        plt.figure(figsize=(8, 8))
        plt.imshow(result['heatmap'], cmap='viridis')
        plt.axis('off')
        plt.title(f"Attention Heatmap - {result['prediction']}")
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Save full analysis
        analyzer.visualize_explanation(np.array(image), result, save_path=analysis_path)
        plt.close('all')  # Close all matplotlib figures to prevent memory leaks

        # Make paths web-accessible
        heatmap_url = "/uploads/" + os.path.basename(heatmap_path)
        analysis_url = "/uploads/" + os.path.basename(analysis_path)

        return JSONResponse({
            "prediction": result['prediction'],
            "confidence": round(result['confidence'] * 100, 2),
            "explanation": {
                "heatmap": heatmap_url,
                "analysis_plot": analysis_url,
                "text": result['explanation']
            }
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

@app.get("/status")
def get_status():
    """Get API status"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "analyzer_initialized": analyzer is not None,
        "device": str(device),
        "version": "5.0 - Smart Explainer"
    }

if __name__ == "__main__":
    import uvicorn
    # Load config to get port
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        port = config.get('api', {}).get('port', 8000)
    except Exception:
        port = 8000
        
    uvicorn.run("api:app", host="127.0.0.1", port=port, reload=True)
