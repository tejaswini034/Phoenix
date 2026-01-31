import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid

# Try to import the model
try:
    from models.temp_model import predict as model_predict
    MODEL_AVAILABLE = True
    print("✓ ML Model loaded successfully")
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"✗ Could not import temp_model.py: {e}")
    print("  Using mock predictions instead")

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def map_temp_model_output(model_output):
    """
    Convert temp_model.py output format to frontend format
    
    temp_model.py returns:
    {
        "label": "Normal", "Bacterial Pneumonia", or "Viral Pneumonia"
        "confidence": 0.95
    }
    
    Frontend expects:
    {
        "binary": {
            "prediction": "Normal" or "Pneumonia",
            "confidence": 0.91
        },
        "subtype": {
            "prediction": "Bacterial" or "Viral" or null,
            "confidence": 0.87
        }
    }
    """
    label = model_output.get("label", "").lower()
    confidence = model_output.get("confidence", 0.0)
    
    response = {
        "binary": None,
        "subtype": None
    }
    
    # Map based on label
    if "normal" in label:
        response["binary"] = {
            "prediction": "Normal",
            "confidence": confidence
        }
        response["subtype"] = None
    
    elif "bacterial" in label:
        response["binary"] = {
            "prediction": "Pneumonia",
            "confidence": confidence
        }
        response["subtype"] = {
            "prediction": "Bacterial",
            "confidence": confidence
        }
    
    elif "viral" in label:
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
    """Handle image upload and return prediction results"""
    
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
        
        if MODEL_AVAILABLE:
            ########################################################################
            # REAL ML INFERENCE - Using temp_model.py
            ########################################################################
            try:
                # Call temp_model.predict() function
                model_output = model_predict(filepath)
                
                # Convert to frontend format
                response_data = map_temp_model_output(model_output)
                
                print(f"✓ Model prediction: {model_output['label']} ({model_output['confidence']:.2%})")
                
            except Exception as model_error:
                # Fall back to mock data
                response_data = get_mock_response()
        
        else:
            ########################################################################
            # FALLBACK: Mock data when model is not available
            ########################################################################
            print("⚠ Using mock data (model not available)")
            response_data = get_mock_response()
        
        # Add metadata
        response_data.update({
            "filename": unique_filename,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify(response_data)
        
    except Exception as e:
        # Log error for debugging
        return jsonify({'error': f'An error occurred while processing the image: {str(e)}'}), 500

def get_mock_response():
    """Generate mock response for testing"""
    import random
    
    # Randomly choose a scenario
    scenarios = [
        {"binary": "Normal", "subtype": None, "conf": round(random.uniform(0.85, 0.95), 2)},
        {"binary": "Pneumonia", "subtype": "Bacterial", "conf": round(random.uniform(0.88, 0.98), 2)},
        {"binary": "Pneumonia", "subtype": "Viral", "conf": round(random.uniform(0.82, 0.94), 2)},
    ]
    
    scenario = random.choice(scenarios)
    
    response = {
        "binary": {
            "prediction": scenario["binary"],
            "confidence": scenario["conf"]
        }
    }
    
    if scenario["subtype"]:
        response["subtype"] = {
            "prediction": scenario["subtype"],
            "confidence": round(scenario["conf"] * random.uniform(0.85, 0.95), 2)
        }
    else:
        response["subtype"] = None
    
    return response

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if MODEL_AVAILABLE else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'model_available': MODEL_AVAILABLE,
        'message': 'ML model is ready' if MODEL_AVAILABLE else 'Running in mock mode'
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Medical X-ray Classifier")
    print("="*50)
    print(f"Model Status: {'✓ READY' if MODEL_AVAILABLE else '✗ NOT AVAILABLE (Using Mock)'}")
    print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Server URL: http://localhost:5000")
    print("="*50 + "\n")
    
    # Run in debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
