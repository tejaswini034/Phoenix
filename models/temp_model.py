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
# CheXNet Medical Model with EXACT Code B Architecture
# =========================================================
class CheXNetMedicalModel(nn.Module):
    """
    CheXNet-based model with EXACT Code B architecture
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # EXACT Code B: Load DenseNet-121 as base_model
        self.base_model = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        )

        # EXACT Code B: Get the feature extractor
        self.features = self.base_model.features

        # EXACT Code B: Get the number of features
        num_features = self.base_model.classifier.in_features

        # EXACT Code B: Store activations and gradients for Grad-CAM
        self.gradients = None
        self.activations = None
        
        # EXACT Code B: Register hooks to the same feature layer
        self.features.register_forward_hook(self.save_activations)
        self.features.register_full_backward_hook(self.save_gradients)

        # EXACT Code B: 6 Pattern detectors with same names
        self.pattern_detectors = nn.ModuleDict({
            'consolidation': self._create_pattern_detector(num_features),
            'ground_glass': self._create_pattern_detector(num_features),
            'pleural_effusion': self._create_pattern_detector(num_features),
            'pulmonary_edema': self._create_pattern_detector(num_features),
            'nodular_opacities': self._create_pattern_detector(num_features),
            'atelectasis': self._create_pattern_detector(num_features),
        })

        # EXACT Code B: Infection extent analyzer with same layers
        self.infection_extent = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, 128),  # EXACT: 128 units
            nn.ReLU(),
            nn.Linear(128, 1),  # EXACT: output single score
            nn.Sigmoid()  # EXACT: Sigmoid activation
        )

        # EXACT Code B: Final classifier with same dimensions
        self.classifier = nn.Sequential(
            nn.Linear(num_features + len(self.pattern_detectors) + 1, 512),  # EXACT: 512 units
            nn.ReLU(),
            nn.Dropout(0.3),  # EXACT: 0.3 dropout
            nn.Linear(512, num_classes)  # EXACT: 3 output classes
        )

    def _create_pattern_detector(self, in_features):
        """EXACT Code B: Pattern detector creation"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, 64),  # EXACT: 64 units
            nn.ReLU(),
            nn.Linear(64, 1),  # EXACT: output single score
            nn.Sigmoid()  # EXACT: Sigmoid activation
        )
    
    def save_activations(self, module, input, output):
        """EXACT Code B: Save activations hook"""
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        """EXACT Code B: Save gradients hook"""
        self.gradients = grad_output[0]

    def forward(self, x):
        """EXACT Code B: Forward pass with same computation order"""
        # EXACT: Extract features
        features = self.features(x)

        # EXACT: Global pooling
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        flattened = pooled.view(pooled.size(0), -1)

        # EXACT: Get pattern scores in dictionary
        pattern_scores = {}
        pattern_list = []

        for name, detector in self.pattern_detectors.items():
            score = detector(features)
            pattern_scores[name] = score
            pattern_list.append(score)

        # EXACT: Get infection extent
        infection_score = self.infection_extent(features)

        # EXACT: Concatenation order: [flattened] + pattern_list + [infection_score]
        combined = torch.cat([flattened] + pattern_list + [infection_score], dim=1)

        # EXACT: Final classification
        output = self.classifier(combined)

        # EXACT Code B returns 4 values
        return output, pattern_scores, infection_score, features

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

    def get_medical_analysis(self, pattern_scores, infection_score, prediction, confidence):
        """Medical analysis matching Code B structure"""
        pattern_names = {
            'consolidation': 'Consolidation',
            'ground_glass': 'Ground-Glass Opacity',
            'pleural_effusion': 'Pleural Effusion',
            'pulmonary_edema': 'Pulmonary Edema',
            'nodular_opacities': 'Nodular Opacities',
            'atelectasis': 'Atelectasis',
        }

        class_names = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']

        analysis = "="*60 + "\n"
        analysis += "MEDICAL PATTERN ANALYSIS\n"
        analysis += "="*60 + "\n\n"

        # Diagnosis
        analysis += f"Diagnosis: {class_names[prediction]}\n"
        analysis += f"Confidence: {confidence:.1%}\n"
        analysis += f"Infection Extent: {infection_score.item():.1%}\n\n"

        # Pattern detection
        analysis += "Detected Patterns:\n"
        significant_patterns = []
        for name, score in pattern_scores.items():
            if score.item() > 0.3:
                significant_patterns.append((pattern_names[name], score.item()))

        if significant_patterns:
            significant_patterns.sort(key=lambda x: x[1], reverse=True)
            for name, score in significant_patterns:
                analysis += f"  • {name}: {score:.1%}\n"
        else:
            analysis += "  • No significant patterns detected\n"

        # Medical interpretation
        analysis += "\nMedical Interpretation:\n"
        if prediction == 1:  # Bacterial
            if pattern_scores['consolidation'].item() > 0.5:
                analysis += "  • Lobar consolidation suggests bacterial pneumonia\n"
            if pattern_scores['pleural_effusion'].item() > 0.3:
                analysis += "  • Pleural effusion may indicate complications\n"

        elif prediction == 2:  # Viral
            if pattern_scores['ground_glass'].item() > 0.5:
                analysis += "  • Ground-glass opacities suggest viral etiology\n"
            if infection_score.item() > 0.5:
                analysis += "  • Extensive involvement typical of viral pneumonia\n"

        return analysis


# =========================================================
# Model Loading
# =========================================================
def load_model(model_path="models/model.pth"):
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Create model with EXACT Code B architecture
    model = CheXNetMedicalModel(num_classes=3)

    # Load checkpoint exactly as Code B does
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract state_dict exactly as Code B does
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Load with strict=True to ensure exact match
    model.load_state_dict(state_dict, strict=True)

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
# Grad-CAM Heatmap Generation (Updated for 4 outputs)
# =========================================================
def generate_gradcam_heatmap(model, input_tensor, original_img, target_class):
    """
    Generate Grad-CAM heatmap and overlay on original image
    """
    try:
        # Forward pass
        model.zero_grad()
        
        # Get model output (now 4 outputs)
        output, _, _, _ = model(input_tensor)
        
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
# Prediction Functions (Updated for 4 outputs)
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
        
        # Forward pass (now returns 4 values)
        model.zero_grad()
        output, pattern_scores, infection_score, _ = model(input_tensor)
        
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
            {k: v[0] for k, v in pattern_scores.items()},
            infection_score[0],
            prediction_idx,
            confidence_val
        )
        
        # Prepare result
        result = {
            "prediction": prediction_idx,
            "prediction_name": class_names[prediction_idx],
            "confidence": confidence_val,
            "probabilities": probs[0].cpu().detach().numpy().tolist(),
            "pattern_scores": {k: v[0].item() for k, v in pattern_scores.items()},
            "infection_score": infection_score[0].item(),
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