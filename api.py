import os
import io
import base64
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from utils.utils import ensure_dir, check_admin_access
import logging

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_PATH = 'models/final_model.h5'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)

# Create upload directory with admin check
if not ensure_dir(UPLOAD_FOLDER):
    raise RuntimeError("Không thể tạo thư mục uploads. Vui lòng chạy lại với quyền admin.")

# Check model directory access
if not check_admin_access('models'):
    raise RuntimeError("Không thể truy cập thư mục models. Vui lòng chạy lại với quyền admin.")

# Load the trained model
model = None
def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Make prediction
def predict_image(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(img)[0][0]
    
    # Determine class and confidence
    is_real = prediction > 0.5
    confidence = float(prediction) if is_real else float(1 - prediction)
    
    result = {
        'is_real': bool(is_real),
        'class': 'Real' if is_real else 'Fake',
        'confidence': round(confidence * 100, 2)
    }
    
    return result

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Check if the file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'File format not supported. Please upload a JPG or PNG image.'}), 400
    
    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Make prediction
        result = predict_image(file_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up the file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for prediction that accepts base64 encoded images.
    
    Request body should be JSON with format:
    {
        "image": "base64_encoded_image_string"
    }
    """
    # Check if request is JSON
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    # Get the image data
    data = request.get_json()
    
    if 'image' not in data:
        return jsonify({'error': 'No image provided in request'}), 400
    
    try:
        # Decode the base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Save the image temporarily
        temp_file = os.path.join(UPLOAD_FOLDER, 'temp_image.jpg')
        image.save(temp_file)
        
        # Make prediction
        result = predict_image(temp_file)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up the file
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)

# Configure logging
log_file = 'app.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)