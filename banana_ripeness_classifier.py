import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. HSV Feature Extraction Module (OpenCV)
# ==========================================

class HSVFeatureExtractor:
    """
    Extracts surface coverage percentages for key color spectrums 
    (Green, Yellow, Brown/Dark spots) associated with banana ripening stages.
    """
    def __init__(self):
        # HSV Color Threshold Ranges in OpenCV (H: 0-180, S: 0-255, V: 0-255)
        self.ranges = {
            'green':  {'lower': np.array([35, 40, 40]),   'upper': np.array([85, 255, 255])},
            'yellow': {'lower': np.array([20, 50, 80]),   'upper': np.array([35, 255, 255])},
            'brown':  {'lower': np.array([0, 20, 20]),    'upper': np.array([20, 200, 150])}
        }

    def extract_features(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Calculates normalized percentage of green, yellow, and brown pixels.
        """
        hsv_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
        
        features = []
        for color, boundaries in self.ranges.items():
            mask = cv2.inRange(hsv_img, boundaries['lower'], boundaries['upper'])
            pixel_count = cv2.countNonZero(mask)
            features.append(pixel_count / float(total_pixels))
            
        return np.array(features, dtype=np.float32)

# ==========================================
# 2. PyTorch Custom Dataset
# ==========================================

class BananaDataset(Dataset):
    """
    Custom Dataset loading images, computing HSV features via OpenCV,
    and applying PyTorch vision transformations.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.hsv_extractor = HSVFeatureExtractor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load with OpenCV for HSV feature extraction
        cv2_img = cv2.imread(img_path)
        if cv2_img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        hsv_feats = self.hsv_extractor.extract_features(cv2_img)

        # Convert BGR to RGB for PyTorch/PIL compatibility
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        if self.transform:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = transforms.ToTensor()(pil_img)

        return tensor_img, torch.tensor(hsv_feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ==========================================
# 3. Hybrid Neural Network Architecture
# ==========================================

class HybridBananaClassifier(nn.Module):
    """
    Combines MobileNetV3-Small deep feature maps with traditional HSV color features.
    """
    def __init__(self, num_classes=5, hsv_feat_dim=3):
        super(HybridBananaClassifier, self).__init__()
        
        # Pretrained MobileNetV3 Backbone
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.cnn_features = backbone.features
        self.avgpool = backbone.avgpool
        
        # MobileNetV3-Small output channels before classifier is 576
        cnn_out_dim = 576
        
        # Classification Head combining CNN + HSV features
        combined_dim = cnn_out_dim + hsv_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_hsv):
        # Extract visual representations via CNN
        feat = self.cnn_features(x_img)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        
        # Concatenate CNN output vectors with raw HSV percentage vector
        combined_feats = torch.cat((feat, x_hsv), dim=1)
        
        # Classify ripeness stage
        out = self.classifier(combined_feats)
        return out

# ==========================================
# 4. Training and Evaluation Loops
# ==========================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, hsv_feats, labels in loader:
        imgs, hsv_feats, labels = imgs.to(device), hsv_feats.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, hsv_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, hsv_feats, labels in loader:
            imgs, hsv_feats, labels = imgs.to(device), hsv_feats.to(device), labels.to(device)

            outputs = model(imgs, hsv_feats)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct.double() / total
    return val_loss, val_acc.item()

# ==========================================
# 5. Main Execution Pipeline
# ==========================================

if __name__ == "__main__":
    # Hyperparameters & Settings
    NUM_CLASSES = 5  # Example: [1: Underripe, 2: Slightly Green, 3: ripe, 4: Very Ripe, 5: Overripe]
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Executing pipeline on device: {DEVICE}")

    # Image Transforms (Careful not to distort natural colors significantly)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dummy Dataset Creation for Demonstration Purpose
    # Replace file paths with paths from Mendeley Banana Ripening Dataset
    dummy_paths = ["banana_sample.jpg"] * 32  
    dummy_labels = [i % NUM_CLASSES for i in range(32)]

    # Create dummy images for script execution check
    for p in set(dummy_paths):
        if not os.path.exists(p):
            cv2.imwrite(p, np.full((224, 224, 3), (0, 255, 255), dtype=np.uint8))

    train_dataset = BananaDataset(dummy_paths, dummy_labels, transform=train_transforms)
    val_dataset = BananaDataset(dummy_paths, dummy_labels, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate Hybrid Model, Loss, and Optimizer
    model = HybridBananaClassifier(num_classes=NUM_CLASSES, hsv_feat_dim=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Training loop
    print("Starting Training Sequence...")
    for epoch in range(NUM_EPOCHS):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_acc = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"| Train Loss: {t_loss:.4f} - Train Acc: {t_acc:.4f} "
              f"| Val Loss: {v_loss:.4f} - Val Acc: {v_acc:.4f}")

    # Export Trained Weights
    torch.save(model.state_dict(), "banana_hybrid_mobilenetv3.pth")
    print("Model successfully saved to 'banana_hybrid_mobilenetv3.pth'.")