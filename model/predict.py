import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Rebuild the 'Skeleton'
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3) # Must match the 3 classes (Cello/Violin/Other)
    
# 2. Load the 'Brain' (the weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load('cello_violin_other_resnet50.pth', map_location=device))
model = model.to(device)

# 3. Set to Evaluation Mode
# This is CRITICAL. It turns off dropout and batch norm updates.
model.eval()

print("Model successfully loaded and ready for prediction!")

def predict_instrument(image_path):
    # 1. Define the same transforms used in training
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load image and convert to RGB
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device) # Add batch dimension: [1, 3, 224, 224]

    # 3. Predict
    with torch.no_grad():
        outputs = model(batch_t)
    print(f"RAW LOGITS: {outputs}")
        
    # 4. Convert output to probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # 5. Get the highest score
    # 0 = Cello, 1 = Other, 2 = Violin (based on how ImageFolder assigns labels)
    class_names = ['Cello', 'Other', 'Violin']
    confidence, index = torch.max(probabilities, 0)
    
    return class_names[index], confidence.item()
