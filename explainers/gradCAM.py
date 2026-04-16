import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa 
from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def gradCAM_explain(input_image_path, original_audio_folder, model_checkpoint, class_names, show_plot=True):
    class_size = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, class_size)
    
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval().to(device)

    # --- 2. IDENTIFY TARGET LAYER ---
    target_layers = [model.layer4[-1]]

    # --- 3. PREPARE THE IMAGE ---
    img = Image.open(input_image_path).convert("RGB").resize((224, 224))
    rgb_img_for_overlay = np.float32(img) / 255.0

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = int(torch.argmax(outputs, dim=1))
        predicted_name = class_names[predicted_idx]

    # --- 4. INITIALIZE AND RUN GRAD-CAM ---
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # --- 5. AUDIO PHYSICS LOOKUP ---
    # Find the original .wav file to get the exact Time and Frequency
    image_filename = os.path.basename(input_image_path)
    target_wav_name = image_filename.replace("_spectrogram.png", ".wav")
    source_dir = original_audio_folder
    original_audio_path = None

    for root, dirs, files in os.walk(source_dir):
        if target_wav_name in files:
            original_audio_path = os.path.join(root, target_wav_name)
            break

    if original_audio_path is None:
        raise FileNotFoundError(f"Could not find the original audio file for {target_wav_name}")

    y, sr = librosa.load(original_audio_path, sr=None)
    exact_duration = librosa.get_duration(y=y, sr=sr)
    nyquist_freq = sr / 2

    # --- 6. VISUALIZE WITH AXES ---
    visualization = show_cam_on_image(rgb_img_for_overlay, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("Original Spectrogram", fontsize=12)

    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM (Predicted: {predicted_name})", fontsize=12)

    # Calculate tick marks for the 224x224 grid
    pixel_ticks = np.linspace(0, 224, 5)
    time_vals = np.linspace(0, exact_duration, 5)
    time_labels = [f"{t:.1f}s" for t in time_vals]
    
    freq_vals = np.linspace(nyquist_freq, 0, 5)
    freq_labels = [f"{int(f)}" for f in freq_vals]

    # Apply the formatted labels to both the original image and the Grad-CAM image
    for ax in axes:
        ax.axis("on") # Ensure axes are turned on
        ax.set_xticks(pixel_ticks)
        ax.set_xticklabels(time_labels, fontsize=8)
        ax.set_xlabel("Time", fontsize=10)
        
        ax.set_yticks(pixel_ticks)
        ax.set_yticklabels(freq_labels, fontsize=8)
        ax.set_ylabel("Frequency (Hz)", fontsize=10)

    plt.tight_layout()

    # --- 7. SAVE AND SHOW ---
    os.makedirs(f"output/gradcam/{predicted_name}", exist_ok=True)
    file_name = image_filename.replace("_spectrogram.png", "_gradcam.png")
    save_path = os.path.join(f"output/gradcam/{predicted_name}", file_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

