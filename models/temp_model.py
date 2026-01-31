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
# CheXNet Medical Model (API-compatible replacement)
# =========================================================
class CheXNetMedicalModel(nn.Module):
    """
    CheXNet-based model that MATCHES SimpleMedicalModel's API
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # DenseNet-121 backbone (CheXNet)
        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        )

        self.features = self.backbone.features
        self.feature_dim = self.backbone.classifier.in_features

        # Pattern detectors (same idea, more of them)
        self.consolidation_detector = self._pattern_head()
        self.glass_opacity_detector = self._pattern_head()

        # Final classifier (keep SAME output behavior)
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

    # KEEP THIS EXACT METHOD NAME
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
# Model Loading (UNCHANGED INTERFACE)
# =========================================================
def load_model(model_path="models/model.pth"):
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = CheXNetMedicalModel(num_classes=3)

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

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
# Grad-CAM (EXTERNAL, SAME AS YOUR WORKING VERSION)
# =========================================================
def generate_heatmap_separate(model, input_tensor, original_img, diagnosis_result):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        # DenseNet last conv layer
        target_layer = model.features[-1]

        class ModelWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out, _, _ = self.m(x)
                return out

        wrapped = ModelWrapper(model)

        cam = GradCAM(
            model=wrapped,
            target_layers=[target_layer]
        )

        targets = [ClassifierOutputTarget(diagnosis_result['prediction'])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        # Normalize original image
        original_norm = original_img.astype(np.float32) / 255.0

        # Resize heatmap
        heatmap = cv2.resize(grayscale_cam, (224, 224))
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET
        )

        # Overlay heatmap on image
        overlay = cv2.addWeighted(
            cv2.cvtColor((original_norm * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            0.6,
            heatmap_color,
            0.4,
            0
        )

        # ---- DEFINE PATH FIRST ----
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        HEATMAP_DIR = os.path.join(BASE_DIR, "static", "heatmaps")
        os.makedirs(HEATMAP_DIR, exist_ok=True)

        name = f"heatmap_{int(time.time()*1000)}.png"
        path = os.path.join(HEATMAP_DIR, name)

        # ---- SAVE OVERLAY ----
        cv2.imwrite(path, overlay)

        print(f"🔥 Heatmap saved at: {path}")

        return overlay, grayscale_cam

    except Exception as e:
        print("❌ Heatmap generation failed")
        print(e)
        import traceback
        traceback.print_exc()
        return None, None


    except Exception as e:
        print(f"Heatmap error: {e}")
        return None, None


# =========================================================
# BACKWARD-COMPATIBILITY FUNCTIONS (CRITICAL)
# =========================================================
def predict(image_path):
    model, class_names = load_model()
    result = predict_with_heatmap(image_path, model, class_names)
    return {
        "label": result["prediction_name"],
        "confidence": result["confidence"]
    }


def predict_with_heatmap(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")
    original_img = np.array(image.resize((224, 224)))

    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)   # IMPORTANT
    print("🧠 Gradients enabled:", input_tensor.requires_grad)
    # 🔥 NO torch.no_grad() here
    output, cons, glass = model(input_tensor)
    probs = F.softmax(output, dim=1)
    confidence, prediction = torch.max(probs, dim=1)

    heatmap_img, _ = generate_heatmap_separate(
        model,
        input_tensor,
        original_img,
        {"prediction": prediction.item()}
    )

    return {
        "prediction": prediction.item(),
        "prediction_name": class_names[prediction.item()],
        "confidence": confidence.item(),
        "heatmap_available": heatmap_img is not None
    }


def predict_image_api(image_path, model_path="models/model.pth"):
    model, class_names = load_model(model_path)
    return predict_with_heatmap(image_path, model, class_names)
