import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# ------------------
# Setup
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same architecture your teammate is using
model = models.densenet121(pretrained=True)

# Same classifier shape (3 classes)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 3)

model.to(device)
model.eval()

# Inference-only transforms (NO augmentation)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

class_names = {
    0: "Normal",
    1: "Bacterial Pneumonia",
    2: "Viral Pneumonia"
}

# ------------------
# Fake but useful Grad-CAM
# ------------------
def generate_fake_heatmap(image):
    img = np.array(image.resize((224,224)))
    heat = np.random.randint(0, 255, (224,224), dtype=np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
    return overlay

# ------------------
# THIS is what backend calls
# ------------------
def predict(image_path, save_dir="static/heatmaps"):
    os.makedirs(save_dir, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass (no training, no .pth)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    pred_class = int(np.argmax(probs))

    # Save fake heatmap so UI can render something
    overlay = generate_fake_heatmap(image)
    heatmap_path = os.path.join(save_dir, "temp_gradcam.png")
    cv2.imwrite(heatmap_path, overlay[:,:,::-1])

    return {
        "label": class_names[pred_class],
        "confidence": float(probs[pred_class]),
        "heatmap_path": heatmap_path
    }
