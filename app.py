import os
import json
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid
from PIL import Image

# Import the model
try:
    from models.temp_model import predict_with_heatmap, load_model
    import torch
    MODEL_AVAILABLE = True
    print("✓ ML Model loaded successfully")
    
    # Load model once at startup
    model, class_names = load_model("models/model.pth")
    print(f"✓ Model loaded with classes: {class_names}")
    
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"✗ Could not import model: {e}")
    print("  Please check temp_model.py exists in models/ folder")
    exit(1)
except Exception as e:
    MODEL_AVAILABLE = False
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['HEATMAP_FOLDER'] = 'static/heatmaps'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def map_model_output(model_output):
    """
    Convert model output format to frontend format
    """
    prediction_name = model_output.get("prediction_name", "").lower()
    confidence = model_output.get("confidence", 0.0)
    
    response = {
        "binary": None,
        "subtype": None,
        "heatmap_url": model_output.get("heatmap_url", None),
        "heatmap_available": model_output.get("heatmap_available", False),
        "probabilities": model_output.get("probabilities", []),
        "consolidation_score": model_output.get("consolidation_score", 0),
        "glass_opacity_score": model_output.get("glass_opacity_score", 0),
        "medical_analysis": model_output.get("medical_analysis", "")
    }
    
    # Map based on label
    if "normal" in prediction_name:
        response["binary"] = {
            "prediction": "Normal",
            "confidence": confidence
        }
        response["subtype"] = None
    
    elif "bacterial" in prediction_name:
        response["binary"] = {
            "prediction": "Pneumonia",
            "confidence": confidence
        }
        response["subtype"] = {
            "prediction": "Bacterial",
            "confidence": confidence
        }
    
    elif "viral" in prediction_name:
        response["binary"] = {
            "prediction": "Pneumonia",
            "confidence": confidence
        }
        response["subtype"] = {
            "prediction": "Viral",
            "confidence": confidence
        }
    
    else:
        # Fallback - treat as pneumonia
        response["binary"] = {
            "prediction": "Pneumonia",
            "confidence": confidence
        }
        response["subtype"] = {
            "prediction": "Other",
            "confidence": confidence
        }
    
    return response

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction results with heatmap"""
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload an image file (PNG, JPG, JPEG, BMP, GIF)'}), 400
    
    try:
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(filepath)
        
        if not MODEL_AVAILABLE or model is None:
            return jsonify({'error': 'ML model is not available. Please check server logs.'}), 500
        
        ########################################################################
        # REAL ML INFERENCE WITH GRAD-CAM
        ########################################################################
        try:
            # Use the predict_with_heatmap function from temp_model
            model_output = predict_with_heatmap(filepath, model, class_names)
            
            # Generate heatmap filename
            heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
            heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
            
            # If heatmap was generated, save it
            if model_output.get("heatmap_available") and model_output.get("heatmap_img") is not None:
                heatmap_img = model_output["heatmap_img"]
                
                # Save heatmap overlay image
                if heatmap_img is not None and len(heatmap_img.shape) == 3:
                    # Convert RGB to BGR for OpenCV save
                    heatmap_bgr = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(heatmap_path, heatmap_bgr)
                    model_output["heatmap_url"] = f"/static/heatmaps/{heatmap_filename}"
                    print(f"✓ Heatmap saved to: {heatmap_path}")
                else:
                    model_output["heatmap_url"] = None
                    print("⚠ Heatmap image format incorrect")
            else:
                model_output["heatmap_url"] = None
                print("⚠ No heatmap generated")
            
            # Convert to frontend format
            response_data = map_model_output(model_output)
            
            print(f"✓ Model prediction: {model_output['prediction_name']} ({model_output['confidence']:.2%})")
            
        except Exception as model_error:
            print(f"❌ Model error: {model_error}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Model prediction failed: {str(model_error)}'}), 500
        
        # Add metadata
        response_data.update({
            "filename": unique_filename,
            "timestamp": datetime.now().isoformat(),
            "uploaded_image_url": f"/uploads/{unique_filename}"
        })
        
        return jsonify(response_data)
        
    except Exception as e:
        # Log error for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An error occurred while processing the image: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_available': MODEL_AVAILABLE,
        'model_classes': class_names if MODEL_AVAILABLE else [],
        'message': 'ML model is ready' if MODEL_AVAILABLE else 'Model not available'
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Medical X-ray Classifier with Grad-CAM")
    print("="*50)
    print(f"Model Status: {'✓ READY' if MODEL_AVAILABLE else '✗ NOT AVAILABLE'}")
    print(f"Model Classes: {class_names if MODEL_AVAILABLE else 'N/A'}")
    print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Heatmap Folder: {app.config['HEATMAP_FOLDER']}")
    print(f"Server URL: http://localhost:5000")
    print("="*50 + "\n")
    
    # Run in debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)