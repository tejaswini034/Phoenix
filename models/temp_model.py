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

# =========================================================
# CheXNet Medical Model with Grad-CAM support
# =========================================================
class CheXNetMedicalModel(nn.Module):
    """
    CheXNet-based model with Grad-CAM support
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # DenseNet-121 backbone (CheXNet)
        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        )

        self.features = self.backbone.features
        self.feature_dim = self.backbone.classifier.in_features
        
        # Store activations and gradients for Grad-CAM
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.features.register_forward_hook(self.save_activations)
        self.features.register_full_backward_hook(self.save_gradients)

        # Pattern detectors
        self.consolidation_detector = self._pattern_head()
        self.glass_opacity_detector = self._pattern_head()

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _pattern_head(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        features = self.features(x)
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        flat = pooled.view(pooled.size(0), -1)

        consolidation_score = self.consolidation_detector(features)
        glass_opacity_score = self.glass_opacity_detector(features)

        combined = torch.cat(
            [flat, consolidation_score, glass_opacity_score], dim=1
        )

        output = self.classifier(combined)

        return output, consolidation_score, glass_opacity_score

    def generate_gradcam(self, target_class=None):
        """Generate Grad-CAM heatmap"""
        if self.activations is None or self.gradients is None:
            return None
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Pool gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        # Average across channels
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # Apply ReLU
        heatmap = F.relu(heatmap)
        
        # Normalize
        if torch.max(heatmap) > 0:
            heatmap = heatmap / torch.max(heatmap)
        
        return heatmap.cpu().detach().numpy()

    def get_medical_analysis(self, consolidation_score, glass_opacity_score, prediction):
        cons = consolidation_score.item()
        glass = glass_opacity_score.item()

        class_names = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
        analysis = "Medical Pattern Analysis:\n"
        analysis += f"• Consolidation score: {cons:.3f}\n"
        analysis += f"• Ground-glass opacity score: {glass:.3f}\n"
        analysis += f"\nDiagnosis: {class_names[prediction]}\n"

        return analysis


# =========================================================
# Model Loading
# =========================================================
def load_model(model_path="models/model.pth"):
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = CheXNetMedicalModel(num_classes=3)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # 🔥 Ignore mismatched classifier layers
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("classifier.")
    }

    model.load_state_dict(filtered_state_dict, strict=False)

    model.to(device)
    model.eval()

    class_names = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
    return model, class_names



# =========================================================
# Image Transform
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================================================
# Grad-CAM Heatmap Generation
# =========================================================
def generate_gradcam_heatmap(model, input_tensor, original_img, target_class):
    """
    Generate Grad-CAM heatmap and overlay on original image
    """
    try:
        # Forward pass
        model.zero_grad()
        
        # Get model output
        output, _, _ = model(input_tensor)
        
        # Calculate gradients
        one_hot = torch.zeros((1, output.shape[1])).to(device)
        one_hot[0][target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Generate heatmap
        heatmap = model.generate_gradcam()
        
        if heatmap is None:
            print("❌ Failed to generate heatmap")
            return None, None
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # Normalize heatmap to 0-255
        heatmap_normalized = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Convert original image to BGR
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        # Overlay heatmap on original image
        superimposed = cv2.addWeighted(
            original_bgr, 0.6,
            heatmap_colored, 0.4, 0
        )
        
        # Convert back to RGB for display
        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        
        return superimposed_rgb, heatmap_resized
        
    except Exception as e:
        print(f"❌ Heatmap generation error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# =========================================================
# Prediction Functions
# =========================================================
def predict(image_path):
    """Simple prediction function"""
    model, class_names = load_model()
    result = predict_with_heatmap(image_path, model, class_names)
    return {
        "prediction_name": result["prediction_name"],
        "confidence": result["confidence"]
    }


def predict_with_heatmap(image_path, model, class_names):
    """Predict with Grad-CAM heatmap generation"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        original_img = np.array(image.resize((224, 224)))
        
        # Transform for model
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Enable gradients
        input_tensor.requires_grad_(True)
        
        # Forward pass
        model.zero_grad()
        output, cons, glass = model(input_tensor)
        
        # Get prediction
        probs = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        
        prediction_idx = prediction.item()
        confidence_val = confidence.item()
        
        # Generate Grad-CAM heatmap
        heatmap_img, raw_heatmap = generate_gradcam_heatmap(
            model, input_tensor, original_img, prediction_idx
        )
        
        # Get medical analysis
        medical_analysis = model.get_medical_analysis(
            cons[0], glass[0], prediction_idx
        )
        
        # Prepare result
        result = {
            "prediction": prediction_idx,
            "prediction_name": class_names[prediction_idx],
            "confidence": confidence_val,
            "probabilities": probs[0].cpu().detach().numpy().tolist(),
            "consolidation_score": cons[0].item(),
            "glass_opacity_score": glass[0].item(),
            "medical_analysis": medical_analysis,
            "heatmap_img": heatmap_img,
            "raw_heatmap": raw_heatmap,
            "heatmap_available": heatmap_img is not None,
            "original_image": original_img
        }
        
        print(f"✓ Prediction: {class_names[prediction_idx]} ({confidence_val:.2%})")
        
        return result
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise


def predict_image_api(image_path, model_path="models/model.pth"):
    """API-compatible prediction function"""
    model, class_names = load_model(model_path)
    return predict_with_heatmap(image_path, model, class_names)