import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# 1. Definir la misma arquitectura utilizada en el entrenamiento
from banana_ripeness_classifier import HybridBananaClassifier, HSVFeatureExtractor

# Cargar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recrear el modelo y cargar los pesos guardados
model = HybridBananaClassifier(num_classes=5).to(device)
model.load_state_dict(torch.load("banana_hybrid_mobilenetv3.pth", map_location=device))
model.eval()

# Definir las mismas transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. Cargar una imagen de prueba
image_path = "banana_ripe2.jpg"
cv2_img = cv2.imread(image_path)
cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

# Extracción de características HSV
extractor = HSVFeatureExtractor()
#hsv_features = torch.tensor(extractor.extract(cv2_img), dtype=torch.float32).unsqueeze(0).to(device)
hsv_features = torch.tensor(extractor.extract_features(cv2_img), dtype=torch.float32).unsqueeze(0).to(device)

# Transformación de imagen para PyTorch
pil_img = Image.fromarray(cv2_img)
img_tensor = transform(pil_img).unsqueeze(0).to(device)

# 3. Predicción
classes = ["Underripe (Inmaduro)", "Slightly Green", "ripe (Maduro)", "Very Ripe (Manchado)", "Overripe (Sobremaduro)"]

with torch.no_grad():
    outputs = model(img_tensor, hsv_features)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

print(f"Estado de Madurez Predicho: {classes[predicted_class]}")
print(f"Confianza: {probabilities[0][predicted_class].item() * 100:.2f}%")