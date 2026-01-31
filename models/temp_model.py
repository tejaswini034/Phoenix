import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2
import time

# ------------------
# Setup
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------
# Simplified Medical Model (from your code)
# ------------------
class SimpleMedicalModel(nn.Module):
    """Simplified model with medical pattern awareness - FIXED VERSION"""

    def __init__(self, num_classes=3):
        super().__init__()

        # Backbone - ResNet18 (lighter)
        backbone = models.resnet18(pretrained=True)

        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Get feature dimensions from the features
        self.feature_dim = 512  # For ResNet18

        # Medical pattern detectors (use same feature space)
        self.consolidation_detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.glass_opacity_detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim + 2, 256),  # 512 + 2 = 514
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # Medical knowledge (fixed parameters)
        self.bacterial_threshold = 0.6
        self.viral_threshold = 0.4

    def forward(self, x):
        # Extract features
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)

        # Detect medical patterns using the same features
        consolidation_score = self.consolidation_detector(features_flat)
        glass_opacity_score = self.glass_opacity_detector(features_flat)

        # Combine features with pattern scores
        combined = torch.cat([features_flat, consolidation_score, glass_opacity_score], dim=1)

        # Final classification
        output = self.classifier(combined)

        return output, consolidation_score, glass_opacity_score

    def get_medical_analysis(self, consolidation_score, glass_opacity_score, prediction):
        """Generate medical explanation"""
        cons_score = consolidation_score.item() if torch.is_tensor(consolidation_score) else consolidation_score
        glass_score = glass_opacity_score.item() if torch.is_tensor(glass_opacity_score) else glass_opacity_score

        analysis = f"Medical Pattern Analysis:\n"
        analysis += f"• Consolidation score: {cons_score:.3f}\n"
        analysis += f"• Ground-glass opacity score: {glass_score:.3f}\n"

        class_names = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
        pred_name = class_names[prediction]

        analysis += f"\nDiagnosis: {pred_name}\n"

        if prediction == 1:  # Bacterial
            if cons_score > self.bacterial_threshold:
                analysis += "✓ Strong consolidation suggests bacterial infection\n"
            if glass_score > 0.3:
                analysis += "⚠️  Some ground-glass opacity present\n"

        elif prediction == 2:  # Viral
            if glass_score > self.viral_threshold:
                analysis += "✓ Ground-glass opacities suggest viral infection\n"
            if cons_score > 0.3:
                analysis += "⚠️  Some consolidation present\n"

        else:  # Normal
            if cons_score < 0.2 and glass_score < 0.2:
                analysis += "✓ No significant patterns detected\n"
            else:
                analysis += "⚠️  Minor patterns present, but within normal limits\n"

        return analysis

# ------------------
# Model Loading Function
# ------------------
# ------------------
# Model Loading Function
# ------------------
def load_model(model_path="models/model.pth"):
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = SimpleMedicalModel(num_classes=3)

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    class_names = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
    return model, class_names
    
# ------------------
# Image Transformations
# ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ------------------
# Heatmap Generation Function (from your logic, modified for separate heatmap)
# ------------------
def generate_heatmap_separate(model, input_tensor, original_img, diagnosis_result):
    """
    Generate Grad-CAM heatmap as a separate image (no overlay)
    """
    try:
        # Get the target layer - different for SimpleMedicalModel vs ResNet
        target_layer = None
        
        if hasattr(model, 'features'):
            # SimpleMedicalModel
            for module in model.features.modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module
        else:
            # Standard ResNet - use layer4
            target_layer = model.layer4[-1].conv2
        
        if target_layer is None:
            print("⚠️ Could not find convolutional layer for heatmap")
            return None, None
        
        # Create Grad-CAM wrapper
        class ModelWrapper(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.model = original_model
            
            def forward(self, x):
                # Handle both model types
                if hasattr(self.model, 'features') and hasattr(self.model, 'consolidation_detector'):
                    output, _, _ = self.model(x)
                else:
                    output = self.model(x)
                return output
        
        # Wrap the model
        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()
        
        # Create Grad-CAM
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            
            cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
            targets = [ClassifierOutputTarget(diagnosis_result['prediction'])]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Convert to color heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            
            print("✓ Heatmap generated successfully")
            
            # Save heatmap
            os.makedirs("static/heatmaps", exist_ok=True)
            timestamp = int(time.time() * 1000)
            heatmap_filename = f"heatmap_{timestamp}.png"
            heatmap_path = os.path.join("static/heatmaps", heatmap_filename)
            cv2.imwrite(heatmap_path, heatmap_resized)
            print(f"✓ Heatmap saved to: {heatmap_path}")
            
            return heatmap_resized, grayscale_cam
            
        except ImportError:
            print("⚠️ Grad-CAM not installed, creating dummy heatmap")
            heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.circle(heatmap, (112, 112), 50, (0, 0, 255), -1)
            
            os.makedirs("static/heatmaps", exist_ok=True)
            timestamp = int(time.time() * 1000)
            heatmap_filename = f"heatmap_{timestamp}.png"
            heatmap_path = os.path.join("static/heatmaps", heatmap_filename)
            cv2.imwrite(heatmap_path, heatmap)
            print(f"✓ Dummy heatmap saved to: {heatmap_path}")
            
            return heatmap, None
            
    except Exception as e:
        print(f"❌ Error generating heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    
# ------------------
# Simple Predict Function (for backward compatibility)
# ------------------
def predict(image_path):
    """
    Simple predict function for backward compatibility
    This is the function that the Flask app is trying to import
    """
    # Load model
    model, class_names = load_model()
    
    if model is None:
        print("❌ Model loading failed")
        return {"label": "Error", "confidence": 0.0}
    
    # Use the existing predict_with_heatmap function
    result = predict_with_heatmap(image_path, model, class_names)
    
    if result is None:
        return {"label": "Error", "confidence": 0.0}
    
    # Return in the expected format
    return {
        "label": result['prediction_name'],
        "confidence": result['confidence']
    }

# ------------------
# Main Prediction Function (with heatmap)
# ------------------
def predict_with_heatmap(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")
    original_img = np.array(image.resize((224, 224)))

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output, consolidation_score, glass_opacity_score = model(input_tensor)
        probs = F.softmax(output, dim=1)

        confidence, prediction = torch.max(probs, dim=1)
        confidence = confidence.item()
        prediction = prediction.item()

        medical_analysis = model.get_medical_analysis(
            consolidation_score[0],
            glass_opacity_score[0],
            prediction
        )

    heatmap_img, _ = generate_heatmap_separate(
        model,
        input_tensor,
        original_img,
        {"prediction": prediction}
    )

    return {
        "prediction": prediction,
        "prediction_name": class_names[prediction],
        "confidence": confidence,
        "probs": probs[0].cpu().numpy().tolist(),
        "consolidation_score": float(consolidation_score[0]),
        "glass_opacity_score": float(glass_opacity_score[0]),
        "analysis": medical_analysis,
        "heatmap_available": heatmap_img is not None
    }
    

# ------------------
# Visualization Function (for testing)
# ------------------
def visualize_results(result):
    """Visualize results with heatmap (for testing)"""
    import matplotlib.pyplot as plt
    
    if result['heatmap_available']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Original image
        axes[0].imshow(result['original_img'])
        axes[0].set_title('Chest X-ray', fontweight='bold', fontsize=12)
        axes[0].axis('off')
        
        # 2. Heatmap image
        axes[1].imshow(result['heatmap_img'])
        axes[1].set_title('Pneumonia Heatmap', fontweight='bold', fontsize=12)
        axes[1].axis('off')
        
        # 3. Confidence bars
        class_names = ['Normal', 'Bacterial', 'Viral']
        probs = result['probs']
        colors = ['green', 'red', 'orange']
        
        bars = axes[2].barh(class_names, probs, color=colors)
        axes[2].set_xlim([0, 1])
        axes[2].set_xlabel('Probability')
        axes[2].set_title('Diagnosis Confidence', fontweight='bold', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # Highlight prediction
        bars[result['prediction']].set_edgecolor('black')
        bars[result['prediction']].set_linewidth(3)
        
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 1. Original image
        axes[0].imshow(result['original_img'])
        axes[0].set_title('Chest X-ray', fontweight='bold', fontsize=12)
        axes[0].axis('off')
        
        # 2. Confidence bars
        class_names = ['Normal', 'Bacterial', 'Viral']
        probs = result['probs']
        colors = ['green', 'red', 'orange']
        
        bars = axes[1].barh(class_names, probs, color=colors)
        axes[1].set_xlim([0, 1])
        axes[1].set_xlabel('Probability')
        axes[1].set_title('Diagnosis Confidence', fontweight='bold', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Highlight prediction
        bars[result['prediction']].set_edgecolor('black')
        bars[result['prediction']].set_linewidth(3)
    
    # Main title
    title = f"Diagnosis: {result['prediction_name']} ({result['confidence']:.1%} confidence)"
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("\n" + "="*70)
    print("MEDICAL ANALYSIS")
    print("="*70)
    print(result['analysis'])

# ------------------
# Test Function
# ------------------
def test_model():
    """Test the model with heatmap generation"""
    print("\n" + "="*70)
    print("Testing Pneumonia Detection Model with Heatmap")
    print("="*70)
    
    # Load the model
    model, class_names = load_model()
    
    if model is None:
        print("❌ Failed to load model")
        return False
    
    # Create a test image
    test_image = "test_xray.jpg"
    if not os.path.exists(test_image):
        from PIL import Image, ImageDraw
        print(f"Creating test image: {test_image}")
        
        # Create a simple X-ray-like image
        img = Image.new('RGB', (512, 512), color='black')
        draw = ImageDraw.Draw(img)
        
        # Draw rib cage
        for i in range(10):
            x = 100 + i * 30
            draw.arc([x, 100, x + 100, 400], 0, 180, fill='white', width=2)
        
        # Draw lungs
        draw.ellipse([150, 150, 200, 300], outline='white', width=2)
        draw.ellipse([300, 150, 350, 300], outline='white', width=2)
        
        img.save(test_image)
        print(f"✓ Created test image: {test_image}")
    
    try:
        # Make prediction with heatmap
        result = predict_with_heatmap(test_image, model, class_names)
        
        if result is None:
            print("❌ Prediction failed")
            return False
        
        print(f"\n✓ Model test successful")
        print(f"✓ Prediction: {result['prediction_name']}")
        print(f"✓ Confidence: {result['confidence']:.2%}")
        print(f"✓ Heatmap generated: {result['heatmap_available']}")
        
        # Show visualization
        visualize_results(result)
        
        print("="*70)
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        print("="*70)
        return False

# ------------------
# Main function for API/backend usage
# ------------------
def predict_image_api(image_path, model_path="models/model.pth"):
    """
    Predict pneumonia for a single image (API-friendly version)
    
    Args:
        image_path: Path to the X-ray image
        model_path: Path to the .pth model file
    
    Returns:
        Dictionary with all results (suitable for JSON response)
    """
    print(f"\nAnalyzing image: {image_path}")
    
    # Load model
    model, class_names = load_model(model_path)
    
    if model is None:
        return {"error": "Model loading failed"}
    
    # Make prediction with heatmap
    result = predict_with_heatmap(image_path, model, class_names)
    
    if result is None:
        return {"error": "Prediction failed"}
    
    # Remove large image arrays from response (they can be converted to base64 if needed)
    result.pop('original_img', None)
    result.pop('heatmap_img', None)
    
    return result

if __name__ == "__main__":
    # Install Grad-CAM if not available
    try:
        import pytorch_grad_cam
    except ImportError:
        print("Installing Grad-CAM...")
        import subprocess
        subprocess.check_call(["pip", "install", "grad-cam"])
    
    # Run test
    test_model()