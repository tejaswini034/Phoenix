import os
import json
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
import base64
from io import BytesIO

# Import the model from temp_model.py
try:
    from models.temp_model import predict_image, get_inference_system
    MODEL_AVAILABLE = True
    print("✓ Medical ML Model loaded successfully")
    
    # Initialize inference system
    inference_system = get_inference_system("models")
    
    # Define class names for both stages
    STAGE1_CLASSES = ["Normal", "Pneumonia"]
    STAGE2_CLASSES = ["Bacterial", "Viral"]
    
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

def convert_heatmap_to_image(heatmap_np, original_image_path):
    """Convert heatmap numpy array to overlay image"""
    if heatmap_np is None:
        return None
    
    try:
        # Load original image
        original_img = Image.open(original_image_path).convert('RGB')
        original_np = np.array(original_img)
        
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap_np, (original_np.shape[1], original_np.shape[0]))
        
        # Convert to 8-bit for OpenCV
        heatmap_8bit = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply medical colormap (hot for better medical visualization)
        heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_HOT)
        
        # Convert from BGR to RGB
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay (30% transparency)
        overlay = cv2.addWeighted(original_np, 0.7, heatmap_rgb, 0.3, 0)
        
        return overlay
        
    except Exception as e:
        print(f"Error creating heatmap overlay: {e}")
        return None

def save_heatmap_image(heatmap_overlay, filename_prefix="heatmap"):
    """Save heatmap overlay image and return URL"""
    if heatmap_overlay is None:
        return None
    
    try:
        # Generate unique filename
        heatmap_filename = f"{filename_prefix}_{uuid.uuid4().hex}.png"
        heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
        
        # Convert RGB to BGR for OpenCV save
        heatmap_bgr = cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, heatmap_bgr)
        
        return f"/static/heatmaps/{heatmap_filename}"
        
    except Exception as e:
        print(f"Error saving heatmap: {e}")
        return None

def map_view_type(view_type_code):
    """Convert view type code to human readable format"""
    view_mapping = {
        0: "Unknown/Other",
        1: "Lateral",
        2: "PA/AP"
    }
    return view_mapping.get(view_type_code, "Unknown")

def generate_medical_analysis(prediction_data):
    """Generate medical analysis text based on prediction data"""
    stage1_pred = prediction_data.get('stage1_prediction')
    stage1_conf = prediction_data.get('stage1_confidence', 0)
    stage2_pred = prediction_data.get('stage2_prediction')
    stage2_conf = prediction_data.get('stage2_confidence', 0)
    final_pred = prediction_data.get('final_prediction')
    
    analysis = []
    
    # Stage 1 analysis
    if stage1_pred == "Normal":
        analysis.append("Normal chest X-ray pattern detected.")
        if stage1_conf >= 0.9:
            analysis.append("High confidence in normal findings.")
        elif stage1_conf >= 0.7:
            analysis.append("Moderate confidence in normal findings.")
        else:
            analysis.append("Low confidence in normal findings.")
    
    elif stage1_pred == "Pneumonia":
        analysis.append("Pneumonia patterns detected in chest X-ray.")
        if stage1_conf >= 0.9:
            analysis.append("High probability of pneumonia.")
        elif stage1_conf >= 0.7:
            analysis.append("Moderate probability of pneumonia.")
        else:
            analysis.append("Low probability of pneumonia.")
        
        # Stage 2 analysis
        if stage2_pred == "Bacterial":
            analysis.append("Pattern suggests bacterial pneumonia.")
            if stage2_conf >= 0.8:
                analysis.append("Strong evidence for bacterial etiology.")
        elif stage2_pred == "Viral":
            analysis.append("Pattern suggests viral pneumonia.")
            if stage2_conf >= 0.8:
                analysis.append("Strong evidence for viral etiology.")
        elif stage2_pred is None:
            analysis.append("Pneumonia type could not be determined.")
    
    # Confidence notes
    if stage1_conf < 0.6:
        analysis.append("Note: Confidence level is low. Clinical correlation required.")
    
    # Final recommendation
    if final_pred == "Normal":
        analysis.append("Recommendation: Routine follow-up.")
    elif "Bacterial" in final_pred:
        analysis.append("Recommendation: Consider antibiotic therapy and clinical evaluation.")
    elif "Viral" in final_pred:
        analysis.append("Recommendation: Supportive care and monitoring advised.")
    else:
        analysis.append("Recommendation: Clinical evaluation recommended.")
    
    return " ".join(analysis)

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
        
        if not MODEL_AVAILABLE or inference_system is None:
            return jsonify({'error': 'ML model is not available. Please check server logs.'}), 500
        
        ########################################################################
        # REAL ML INFERENCE WITH TWO-STAGE CLASSIFICATION
        ########################################################################
        try:
            # Get prediction using the new medical inference system
            prediction_data = predict_image(filepath, "models")
            
            # Check for errors
            if prediction_data.get('error'):
                return jsonify({'error': f'Model prediction failed: {prediction_data["error"]}'}), 500
            
            # Convert heatmaps to overlay images
            heatmap1_overlay = convert_heatmap_to_image(prediction_data.get('heatmap_stage1'), filepath)
            heatmap2_overlay = convert_heatmap_to_image(prediction_data.get('heatmap_stage2'), filepath)
            
            # Save heatmap images and get URLs
            heatmap1_url = save_heatmap_image(heatmap1_overlay, "heatmap_stage1") if heatmap1_overlay is not None else None
            heatmap2_url = save_heatmap_image(heatmap2_overlay, "heatmap_stage2") if heatmap2_overlay is not None else None
            
            # Prepare probabilities array
            probabilities = []
            
            # Stage 1 probabilities
            stage1_conf = prediction_data.get('stage1_confidence', 0)
            if prediction_data.get('stage1_prediction') == "Normal":
                probabilities.append({
                    "class": "Normal",
                    "probability": stage1_conf,
                    "stage": 1
                })
                probabilities.append({
                    "class": "Pneumonia",
                    "probability": 1 - stage1_conf,
                    "stage": 1
                })
            else:  # Pneumonia
                probabilities.append({
                    "class": "Normal",
                    "probability": 1 - stage1_conf,
                    "stage": 1
                })
                probabilities.append({
                    "class": "Pneumonia",
                    "probability": stage1_conf,
                    "stage": 1
                })
            
            # Stage 2 probabilities
            stage2_pred = prediction_data.get('stage2_prediction')
            stage2_conf = prediction_data.get('stage2_confidence')
            
            if stage2_pred:
                if stage2_pred == "Bacterial":
                    probabilities.append({
                        "class": "Bacterial Pneumonia",
                        "probability": stage2_conf,
                        "stage": 2
                    })
                    probabilities.append({
                        "class": "Viral Pneumonia",
                        "probability": 1 - stage2_conf,
                        "stage": 2
                    })
                elif stage2_pred == "Viral":
                    probabilities.append({
                        "class": "Viral Pneumonia",
                        "probability": stage2_conf,
                        "stage": 2
                    })
                    probabilities.append({
                        "class": "Bacterial Pneumonia",
                        "probability": 1 - stage2_conf,
                        "stage": 2
                    })
            
            # Generate medical analysis
            medical_analysis = generate_medical_analysis(prediction_data)
            
            # Prepare response in the format expected by frontend
            response_data = {
                "binary": {
                    "prediction": prediction_data.get('stage1_prediction'),
                    "confidence": float(prediction_data.get('stage1_confidence', 0))
                },
                "subtype": None,
                "heatmap_url": heatmap1_url,  # Use stage1 heatmap as main heatmap
                "heatmap_available": heatmap1_url is not None,
                "stage2_heatmap_url": heatmap2_url,
                "stage2_heatmap_available": heatmap2_url is not None,
                "probabilities": probabilities,
                "medical_analysis": medical_analysis,
                "consolidation_score": 0,  # Not used in new model
                "glass_opacity_score": 0,   # Not used in new model
                "view_type": map_view_type(prediction_data.get('view_type', 0)),
                "final_prediction": prediction_data.get('final_prediction')
            }
            
            # Add subtype information if available
            if prediction_data.get('stage2_prediction'):
                response_data["subtype"] = {
                    "prediction": prediction_data.get('stage2_prediction'),
                    "confidence": float(prediction_data.get('stage2_confidence', 0))
                }
            
            print(f"✓ Medical analysis complete:")
            print(f"  Stage 1: {prediction_data['stage1_prediction']} ({prediction_data['stage1_confidence']:.2%})")
            if prediction_data.get('stage2_prediction'):
                print(f"  Stage 2: {prediction_data['stage2_prediction']} ({prediction_data['stage2_confidence']:.2%})")
            print(f"  Final: {prediction_data['final_prediction']}")
            
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
        'model_type': 'Medical Two-Stage ResNet-18 with Grad-CAM',
        'stage1_classes': STAGE1_CLASSES if MODEL_AVAILABLE else [],
        'stage2_classes': STAGE2_CLASSES if MODEL_AVAILABLE else [],
        'message': 'Medical ML model is ready' if MODEL_AVAILABLE else 'Model not available'
    })

# Add predict_with_heatmap function for backward compatibility
def predict_with_heatmap(image_path, model=None, class_names=None):
    """Wrapper function for backward compatibility with app.py"""
    if not MODEL_AVAILABLE:
        return {
            "prediction_name": "Error",
            "confidence": 0.0,
            "heatmap_available": False,
            "heatmap_img": None,
            "probabilities": [],
            "medical_analysis": "Model not available"
        }
    
    try:
        # Get prediction using the new system
        prediction_data = predict_image(image_path, "models")
        
        if prediction_data.get('error'):
            return {
                "prediction_name": "Error",
                "confidence": 0.0,
                "heatmap_available": False,
                "heatmap_img": None,
                "probabilities": [],
                "medical_analysis": prediction_data.get('error')
            }
        
        # Convert heatmap to overlay image
        heatmap_overlay = convert_heatmap_to_image(prediction_data.get('heatmap_stage1'), image_path)
        
        # Determine prediction name for backward compatibility
        final_pred = prediction_data.get('final_prediction', 'Unknown')
        if final_pred == "Normal":
            pred_name = "Normal"
        elif final_pred == "Bacterial Pneumonia":
            pred_name = "Bacterial Pneumonia"
        elif final_pred == "Viral Pneumonia":
            pred_name = "Viral Pneumonia"
        else:
            pred_name = "Pneumonia"
        
        return {
            "prediction_name": pred_name,
            "confidence": float(prediction_data.get('stage1_confidence', 0)),
            "heatmap_available": heatmap_overlay is not None,
            "heatmap_img": heatmap_overlay,
            "probabilities": [],  # Will be populated by app.py
            "medical_analysis": generate_medical_analysis(prediction_data),
            "consolidation_score": 0,
            "glass_opacity_score": 0
        }
        
    except Exception as e:
        print(f"Error in predict_with_heatmap: {e}")
        return {
            "prediction_name": "Error",
            "confidence": 0.0,
            "heatmap_available": False,
            "heatmap_img": None,
            "probabilities": [],
            "medical_analysis": f"Prediction error: {str(e)}"
        }

# Dummy load_model function for backward compatibility
def load_model(model_path):
    """Dummy function for backward compatibility"""
    return None, ["Normal", "Pneumonia", "Bacterial", "Viral"]

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Medical X-ray Classifier with Two-Stage Grad-CAM")
    print("="*50)
    print(f"Model Status: {'✓ READY' if MODEL_AVAILABLE else '✗ NOT AVAILABLE'}")
    print(f"Stage 1 Classes: {STAGE1_CLASSES}")
    print(f"Stage 2 Classes: {STAGE2_CLASSES}")
    print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Heatmap Folder: {app.config['HEATMAP_FOLDER']}")
    print(f"Server URL: http://localhost:5000")
    print("="*50 + "\n")
    
    # Run in debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)