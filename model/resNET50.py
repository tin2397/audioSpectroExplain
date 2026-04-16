import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


"""
This ResNet50 model is designed to classify spectrogram images into three categories: Cello, Violin, and Other.
The model architecture is based on the standard ResNet50, but the final fully connected layer has been modified to output three classes instead of the original 1000.
The model is trained on a dataset of spectrogram images generated from audio files, which are organized into subfolders based on their instrument class. 
The training process includes data loading, splitting into training and validation sets, and a standard training loop with loss calculation and backpropagation.

IF WANT TO TRAIN A NEW MODEL:
1. Ensure your spectrogram images are organized in a folder structure like:
   - spectrograms_output/
     - Cello/
     - Violin/
     - Other/
2. CHECK model.fc in the code below to match the number of classes we have (2 for Cello vs Violin, 3 if you include Other).
"""


# 1. Define Transforms (Matching ResNet's expectations)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Load all data from the root folder
full_dataset = datasets.ImageFolder(root=r'C:\Users\tin23\OneDrive\Desktop\HONS WORKING\playWithSHAP\spectrograms_output\Penguin', transform=data_transforms)

# Extract the labels (0 for Cello, 1 for Violin, etc.) so we can balance them
targets = full_dataset.targets
indices = list(range(len(full_dataset)))


# 3. The Split: 80% for Training, 20% for Validation
# 'stratify=targets' ensures Cello, Violin, and Other are split equally!
train_idx, val_idx, train_targets, val_targets = train_test_split(
    indices, targets, test_size=0.20, stratify=targets, random_state=42
)

# 4. Create the final PyTorch Datasets
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

print(f"Total images loaded: {len(full_dataset)}")
print(f"Images for Training (80%): {len(train_dataset)}")
print(f"Images for Validation (20%): {len(val_dataset)}")

val_counts = {0: 0, 1: 0}
for _, label in val_dataset:
    val_counts[label] += 1

print(f"Validation Set Breakdown: Penguin 13B3-1={val_counts[0]},Penguin 14B19-1={val_counts[1]}")
# 4. Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Ready! Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

# Setup Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2) # Cello vs Violin
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- THE TRAINING LOOP ---
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass (Learning)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Validation Phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Accuracy: {100 * correct / total:.2f}%")


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trained_models_dir = os.path.join(project_root, "trained_models")
os.makedirs(trained_models_dir, exist_ok=True)

model_output_path = os.path.join(trained_models_dir, "penguin_resnet50.pth")
torch.save(model.state_dict(), model_output_path)
print(f"Model saved to: {model_output_path}")