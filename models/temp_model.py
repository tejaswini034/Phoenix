import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

class MedicalPneumoniaClassifier(nn.Module):
    """Medical-grade classifier with attention mechanism"""
    
    def __init__(self, num_classes: int = 2, use_view: bool = True):
        super(MedicalPneumoniaClassifier, self).__init__()
        self.use_view = use_view
        
        # ResNet-18 backbone
        self.backbone = models.resnet18(pretrained=False)
        
        # Extract features before final layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Adaptive pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Attention mechanism for medical regions
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        
        # Classifier
        classifier_input = 512 + 1 if use_view else 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(classifier_input, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor, view_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with medical attention"""
        # Extract features
        features = self.features(x)
        
        # Apply attention
        batch_size = features.size(0)
        spatial_features = features.view(batch_size, 512, -1).mean(dim=2)
        attention_weights = self.attention(spatial_features).view(batch_size, 512, 1, 1)
        features = features * attention_weights
        
        # Pool features
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        
        # Add view information if provided
        if self.use_view and view_type is not None:
            if view_type.dim() == 1:
                view_type = view_type.unsqueeze(1)
            features = torch.cat([features, view_type.float()], dim=1)
        
        # Classify
        return self.classifier(features)
    
    def get_last_conv(self):
        """Get the last convolutional layer for Grad-CAM"""
        return self.backbone.layer4[-1]

class PneumoniaInferenceSystem:
    """Complete pneumonia inference system with two-stage classification"""
    
    def __init__(self, model_dir: str = "models", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        
        # Initialize models
        self.stage1_model = None
        self.stage2_model = None
        
        # Grad-CAM storage
        self.activations = {}
        self.gradients = {}
        
        # Medical configuration
        self.medical_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load models
        self._load_models()
        
        # Register hooks for Grad-CAM
        self._register_hooks()
    
    def _load_medical_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load medical checkpoint safely"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            return checkpoint
        except:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                return checkpoint
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")
    
    def _load_models(self):
        """Load both stage models"""
        try:
            # Load Stage 1 model
            stage1_path = f"{self.model_dir}/stage1_medical_best.pth"
            stage1_checkpoint = self._load_medical_checkpoint(stage1_path)
            
            self.stage1_model = MedicalPneumoniaClassifier(num_classes=2, use_view=True)
            self.stage1_model.load_state_dict(stage1_checkpoint['model_state_dict'])
            self.stage1_model.to(self.device)
            self.stage1_model.eval()
            
            # Load Stage 2 model
            stage2_path = f"{self.model_dir}/stage2_medical_best.pth"
            stage2_checkpoint = self._load_medical_checkpoint(stage2_path)
            
            self.stage2_model = MedicalPneumoniaClassifier(num_classes=2, use_view=True)
            self.stage2_model.load_state_dict(stage2_checkpoint['model_state_dict'])
            self.stage2_model.to(self.device)
            self.stage2_model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}")
    
    def _register_hooks(self):
        """Register hooks for Grad-CAM"""
        def save_activation_stage1(module, input, output):
            self.activations['stage1'] = output.detach()
        
        def save_gradient_stage1(module, grad_input, grad_output):
            self.gradients['stage1'] = grad_output[0].detach()
        
        def save_activation_stage2(module, input, output):
            self.activations['stage2'] = output.detach()
        
        def save_gradient_stage2(module, grad_input, grad_output):
            self.gradients['stage2'] = grad_output[0].detach()
        
        # Register hooks on last conv layers
        target_layer1 = self.stage1_model.get_last_conv()
        target_layer2 = self.stage2_model.get_last_conv()
        
        target_layer1.register_forward_hook(save_activation_stage1)
        target_layer1.register_full_backward_hook(save_gradient_stage1)
        target_layer2.register_forward_hook(save_activation_stage2)
        target_layer2.register_full_backward_hook(save_gradient_stage2)
    
    def _detect_view_type(self, image_path: str) -> int:
        """Detect medical view type from image aspect ratio"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                aspect_ratio = width / height if height > 0 else 1.0
                
                if aspect_ratio < 0.85:
                    return 1  # Lateral view
                elif aspect_ratio > 1.2:
                    return 2  # PA/AP view
                else:
                    return 0  # Unknown/other
        except:
            return 0
    
    def _generate_gradcam(self, model_name: str, image_tensor: torch.Tensor, 
                         view_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Generate Grad-CAM heatmap for the specified model"""
        torch.backends.cudnn.deterministic = True
        
        # Get model
        model = self.stage1_model if model_name == 'stage1' else self.stage2_model
        
        # Prepare inputs
        if view_tensor.dim() == 1:
            view_tensor = view_tensor.unsqueeze(0)
        view_tensor = view_tensor.to(self.device)
        
        # Forward pass
        model.zero_grad()
        image_tensor_grad = image_tensor.unsqueeze(0).to(self.device).requires_grad_(True)
        output = model(image_tensor_grad, view_tensor)
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=False)
        
        # Get gradients and activations
        gradients = self.gradients.get(model_name)
        activations = self.activations.get(model_name)
        
        if gradients is None or activations is None:
            return np.zeros((7, 7))
        
        # Process Grad-CAM
        gradients = gradients.cpu()
        activations = activations.cpu()
        
        # Pool gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activations
        weighted_activations = torch.zeros_like(activations)
        for i in range(activations.size(1)):
            weighted_activations[:, i, :, :] = activations[:, i, :, :] * pooled_gradients[i]
        
        # Generate heatmap
        heatmap = torch.mean(weighted_activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap.numpy()
        
        # Normalize
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, int]:
        """Preprocess image for inference"""
        # Load and convert to RGB
        image = Image.open(image_path).convert('RGB')
        original_np = np.array(image)
        
        # Apply transforms
        image_tensor = self.medical_transform(image)
        
        # Detect view type
        view_type = self._detect_view_type(image_path)
        
        return image_tensor, original_np, view_type
    
    def predict_image(self, image_path: str) -> Dict:
        """
        Main inference function.
        
        Returns:
            Dictionary with:
            - stage1_prediction: "Normal" or "Pneumonia"
            - stage1_confidence: float
            - stage2_prediction: "Bacterial", "Viral", or None
            - stage2_confidence: float or None
            - final_prediction: "Normal", "Bacterial Pneumonia", or "Viral Pneumonia"
            - heatmap_stage1: numpy array or None
            - heatmap_stage2: numpy array or None
        """
        try:
            # Preprocess image
            image_tensor, original_np, view_type = self.preprocess_image(image_path)
            view_tensor = torch.tensor([[view_type]], dtype=torch.float32).to(self.device)
            
            # Stage 1: Normal vs Pneumonia
            with torch.no_grad():
                stage1_output = self.stage1_model(
                    image_tensor.unsqueeze(0).to(self.device), 
                    view_tensor
                )
                stage1_probs = F.softmax(stage1_output, dim=1)
                pneumonia_prob = stage1_probs[0, 1].item()
                normal_prob = stage1_probs[0, 0].item()
            
            stage1_pred = "Pneumonia" if pneumonia_prob > 0.5 else "Normal"
            stage1_confidence = pneumonia_prob if stage1_pred == "Pneumonia" else normal_prob
            
            # Generate Stage 1 heatmap
            target_class = 1 if stage1_pred == "Pneumonia" else 0
            heatmap_stage1 = self._generate_gradcam(
                'stage1', image_tensor, view_tensor, target_class
            )
            
            # Initialize stage2 results
            stage2_pred = None
            stage2_confidence = None
            heatmap_stage2 = None
            
            # Stage 2: Bacterial vs Viral (only if pneumonia detected)
            if stage1_pred == "Pneumonia" and pneumonia_prob > 0.5:
                with torch.no_grad():
                    stage2_output = self.stage2_model(
                        image_tensor.unsqueeze(0).to(self.device),
                        view_tensor
                    )
                    stage2_probs = F.softmax(stage2_output, dim=1)
                    
                    if stage2_probs[0, 0] > stage2_probs[0, 1]:
                        stage2_pred = "Bacterial"
                        stage2_confidence = stage2_probs[0, 0].item()
                        target_class = 0
                    else:
                        stage2_pred = "Viral"
                        stage2_confidence = stage2_probs[0, 1].item()
                        target_class = 1
                
                # Generate Stage 2 heatmap
                heatmap_stage2 = self._generate_gradcam(
                    'stage2', image_tensor, view_tensor, target_class
                )
            
            # Determine final prediction
            if stage1_pred == "Normal":
                final_pred = "Normal"
            elif stage2_pred == "Bacterial":
                final_pred = "Bacterial Pneumonia"
            elif stage2_pred == "Viral":
                final_pred = "Viral Pneumonia"
            else:
                final_pred = "Pneumonia (type unknown)"
            
            # Resize heatmaps to match original image dimensions
            if heatmap_stage1 is not None:
                heatmap_stage1 = cv2.resize(
                    heatmap_stage1, 
                    (original_np.shape[1], original_np.shape[0])
                )
            
            if heatmap_stage2 is not None:
                heatmap_stage2 = cv2.resize(
                    heatmap_stage2,
                    (original_np.shape[1], original_np.shape[0])
                )
            
            return {
                'stage1_prediction': stage1_pred,
                'stage1_confidence': float(stage1_confidence),
                'stage2_prediction': stage2_pred,
                'stage2_confidence': float(stage2_confidence) if stage2_confidence else None,
                'final_prediction': final_pred,
                'heatmap_stage1': heatmap_stage1,
                'heatmap_stage2': heatmap_stage2,
                'view_type': view_type
            }
            
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            return {
                'stage1_prediction': "Error",
                'stage1_confidence': 0.0,
                'stage2_prediction': None,
                'stage2_confidence': None,
                'final_prediction': "Error",
                'heatmap_stage1': None,
                'heatmap_stage2': None,
                'view_type': 0,
                'error': str(e)
            }

# Global instance for easy import
_inference_system = None

def get_inference_system(model_dir: str = "models") -> PneumoniaInferenceSystem:
    """Get or create the inference system singleton"""
    global _inference_system
    if _inference_system is None:
        _inference_system = PneumoniaInferenceSystem(model_dir)
    return _inference_system

def predict_image(image_path: str, model_dir: str = "models") -> Dict:
    """
    Public function for image prediction.
    
    Args:
        image_path: Path to the input image
        model_dir: Directory containing the trained models
    
    Returns:
        Dictionary with prediction results
    """
    system = get_inference_system(model_dir)
    return system.predict_image(image_path)

def predict_with_heatmap(image_path: str, model_dir: str = "models") -> Dict:
    """
    Alias for predict_image for backward compatibility.
    This function is referenced in app.py.
    """
    return predict_image(image_path, model_dir)


# For backward compatibility
if __name__ == "__main__":
    # Example usage
    result = predict_image("test_image.jpg")
    print(result)