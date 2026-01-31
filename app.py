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
    print("✓ ML Model loaded successfully from temp_model.py")
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
        "confidence": 0.95,
        "heatmap_path": "/static/heatmaps/temp_gradcam.png"
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
    heatmap_path = model_output.get("heatmap_path", "")
    
    response = {
        "binary": None,
        "subtype": None,
        "heatmap_path": heatmap_path
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
                model_output = model_predict(filepath, save_dir="static/heatmaps")
                
                # Convert to frontend format
                response_data = map_temp_model_output(model_output)
                
                print(f"✓ Model prediction: {model_output['label']} ({model_output['confidence']:.2%})")
                if model_output.get("heatmap_path"):
                    print(f"✓ Heatmap saved at: {model_output['heatmap_path']}")
                
            except Exception as model_error:
                app.logger.error(f"Model prediction error: {str(model_error)}")
                # Fall back to mock data
                response_data = get_mock_response(unique_filename, with_heatmap=True)
                response_data['model_error'] = str(model_error)
        
        else:
            ########################################################################
            # FALLBACK: Mock data when model is not available
            ########################################################################
            print("⚠ Using mock data (model not available)")
            response_data = get_mock_response(unique_filename, with_heatmap=True)
        
        # Add metadata
        response_data.update({
            "filename": unique_filename,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify(response_data)
        
    except Exception as e:
        # Log error for debugging
        app.logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': f'An error occurred while processing the image: {str(e)}'}), 500

def get_mock_response(filename, with_heatmap=True):
    """Generate mock response for testing"""
    import random
    
    # Randomly choose a scenario
    scenarios = [
        {"binary": "Normal", "subtype": None, "conf": 0.88},
        {"binary": "Pneumonia", "subtype": "Bacterial", "conf": 0.91},
        {"binary": "Pneumonia", "subtype": "Viral", "conf": 0.87},
        {"binary": "Pneumonia", "subtype": "Bacterial", "conf": 0.94},
        {"binary": "Normal", "subtype": None, "conf": 0.82}
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
            "confidence": scenario["conf"] * 0.95  # Slightly lower for subtype
        }
    else:
        response["subtype"] = None
    
    # Add heatmap path if requested
    if with_heatmap:
        # Check if test heatmap exists
        test_heatmap = "static/heatmaps/test_heatmap.png"
        if os.path.exists(test_heatmap):
            response["heatmap_path"] = "/" + test_heatmap
        else:
            # Create a simple test heatmap if it doesn't exist
            try:
                create_test_heatmap()
                response["heatmap_path"] = "/static/heatmaps/test_heatmap.png"
            except:
                response["heatmap_path"] = ""
    
    return response

def create_test_heatmap():
    """Create a test heatmap image if it doesn't exist"""
    from PIL import Image, ImageDraw
    import numpy as np
    
    heatmap_dir = "static/heatmaps"
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Create a simple gradient heatmap
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Create a radial gradient (red in center, blue at edges)
    for x in range(400):
        for y in range(300):
            # Calculate distance from center (simulating lung area)
            dx = x - 200
            dy = y - 150
            distance = np.sqrt(dx*dx + dy*dy)
            max_distance = np.sqrt(200**2 + 150**2)
            
            # Normalize distance
            normalized = distance / max_distance
            
            # Red intensity (high in center - simulating pathology focus)
            red = int(255 * (1 - normalized))
            # Blue intensity (high at edges)
            blue = int(255 * normalized)
            
            # Set pixel with some transparency
            draw.point((x, y), fill=(red, 50, blue, 150))
    
    img.save(os.path.join(heatmap_dir, 'test_heatmap.png'))
    print(f"✓ Created test heatmap at: {heatmap_dir}/test_heatmap.png")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy' if MODEL_AVAILABLE else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'model_available': MODEL_AVAILABLE,
        'message': 'ML model is ready' if MODEL_AVAILABLE else 'Running in mock mode'
    }
    
    # Check if heatmap directory exists
    if os.path.exists('static/heatmaps'):
        health_status['heatmap_dir'] = 'available'
        heatmap_files = os.listdir('static/heatmaps')
        health_status['heatmap_files'] = heatmap_files
    else:
        health_status['heatmap_dir'] = 'not_available'
    
    return jsonify(health_status)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/heatmaps', exist_ok=True)
    
    print("\n" + "="*50)
    print("Medical X-ray Classifier Starting...")
    print("="*50)
    print(f"Model available: {'✓ YES' if MODEL_AVAILABLE else '✗ NO (using mock data)'}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Heatmap folder: static/heatmaps")
    print(f"Server URL: http://localhost:5000")
    print("="*50 + "\n")
    
    # Run in debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5000)