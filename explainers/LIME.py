import torch
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
from torchvision import models, transforms

# 1. SETUP MODEL & DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('cello_violin_NEW_resnet50.pth', map_location=device))
model.to(device)
model.eval()

# 2. MATCH TRAINING TRANSFORMS
# Note: LIME provides images as NumPy arrays [0, 255], 
# so we need to convert them to Tensors and normalize inside the function.
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. THE BRIDGE FUNCTION
def batch_predict(images):
    """
    LIME feeds this function a batch of NumPy images.
    We return a batch of probabilities.
    """
    model.eval()
    batch = torch.stack([preprocess(i) for i in images], dim=0)
    batch = batch.to(device)
    
    with torch.no_grad():
        logits = model(batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

# 4. RUN LIME
def run_lime_explanation(test_image_path):
    # 1. GENERATE THE LIME EXPLANATION
    img = Image.open(test_image_path).convert('RGB')
    img_array = np.array(img.resize((224, 224)))
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array, 
        batch_predict, # Using the function from our previous snippet
        top_labels=2, 
        num_samples=1000
    )

    # 2. FIND ORIGINAL AUDIO PHYSICS
    image_filename = os.path.basename(test_image_path)
    target_wav_name = image_filename.replace("_spectrogram.png", ".wav")
    source_dir = "dataset/TinySOL/TinySOL2020"
    
    original_audio_path = None
    for root, dirs, files in os.walk(source_dir):
        if target_wav_name in files:
            original_audio_path = os.path.join(root, target_wav_name)
            break 

    # Default values in case audio isn't found
    exact_duration = 4.0 
    nyquist_freq = 22050

    if original_audio_path:
        y, sr = librosa.load(original_audio_path, sr=None)
        exact_duration = librosa.get_duration(y=y, sr=sr)
        nyquist_freq = sr / 2

    # 3. VISUALIZE WITH AXES
    top_class = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(top_class, positive_only=True, num_features=5, hide_rest=False)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # Show the LIME mask
    ax.imshow(mark_boundaries(temp, mask))
    ax.axis('on') # Force axes to appear

    # --- APPLY PHYSICS LABELS ---
    pixel_ticks = np.linspace(0, 224, 5)
    class_names = ['Cello', 'Other', 'Violin']
    # Time Labels (X-axis)
    time_labels = [f"{t:.1f}s" for t in np.linspace(0, exact_duration, 5)]
    ax.set_xticks(pixel_ticks)
    ax.set_xticklabels(time_labels)
    ax.set_xlabel('Time')

    # Frequency Labels (Y-axis) - Note: 0 pixel is top (Nyquist), 224 is bottom (0Hz)
    freq_labels = [f"{int(f)}" for f in np.linspace(nyquist_freq, 0, 5)]
    ax.set_yticks(pixel_ticks)
    ax.set_yticklabels(freq_labels)
    ax.set_ylabel('Frequency (Hz)')

    plt.title(f"LIME Explanation for {class_names[top_class]}")
    plt.tight_layout()
    plt.show()
