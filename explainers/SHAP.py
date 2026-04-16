import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import shap
import matplotlib.pyplot as plt
import librosa
import os


def shap_explain(input_image_path, full_spectrogram_folder_for_background,original_audio_folder, model_checkpoint, class_names, show_plot=True):
    # --- 1. SETUP MODEL ---
    class_size = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, class_size)


    # Rebuild the skeleton and load the brain

    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model = model.to(device)
    model.eval() # Critical for SHAP

    # --- 2. SETUP DATA TRANFORMS ---
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- 3. CREATE THE "BACKGROUND" ---
    print("Loading background dataset for SHAP...")
    full_dataset = datasets.ImageFolder(root=full_spectrogram_folder_for_background, transform=preprocess)
    background_loader = DataLoader(full_dataset, batch_size=50, shuffle=True)
    background_images, _ = next(iter(background_loader))
    background_images = background_images.to(device)

    # --- 4. INITIALIZE SHAP ---
    print("Initializing GradientExplainer...")
    explainer = shap.GradientExplainer(model, background_images)

    # --- 5. PREPARE THE TEST IMAGE ---
    test_image_path = input_image_path

    img = Image.open(test_image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device) # Shape is [1, 3, 224, 224]

    # --- 6. CALCULATE SHAP VALUES ---
    print("Calculating SHAP values (this might take a few seconds)...")
    shap_values, indexes = explainer.shap_values(img_tensor, ranked_outputs=1)

    predicted_class_idx = indexes[0][0].item()
    predicted_name = class_names[predicted_class_idx]
    print(f"Model Predicted: {predicted_name}")

    # --- 7. THE SELF-AWARE FORMAT AND PLOT ---
    print("Formatting arrays for plotting...")

    # 1. Format the Image Tensor
    test_numpy = img_tensor.cpu().numpy()
    if test_numpy.shape[1] == 3: 
        test_numpy = np.transpose(test_numpy, (0, 2, 3, 1))
        # print(test_numpy.shape)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    test_numpy = test_numpy * std + mean
    test_numpy = np.clip(test_numpy, 0, 1)

    # 2. Format the SHAP Tensor
    shap_array = shap_values[0]
    if shap_array.shape[0] == 3:
        shap_array = np.transpose(shap_array, (3, 1, 2, 0))
    # 3. Generate the plot, but KEEP IT HIDDEN (show=False)
    #  (# samples x width x height x channels) for shap array and test_numpy
    shap.image_plot([shap_array], test_numpy, labels=np.array([[predicted_name]]), show=False)

    print("Calculating exact physics for axes...")

    image_filename = os.path.basename(test_image_path)
    # Figure out where the original audio file is by reversing the name
    target_wav_name = image_filename.replace("_spectrogram.png", ".wav")

    source_dir = original_audio_folder

    for root, dirs, files in os.walk(source_dir):
        if target_wav_name in files:
            original_audio_path = os.path.join(root, target_wav_name)
            break 
    try:
        # Read the real physics of the audio file
        y, sr = librosa.load(original_audio_path, sr=None)
        exact_duration = librosa.get_duration(y=y, sr=sr)
        nyquist_freq = sr / 2  # The highest frequency in a spectrogram is half the sample rate
        
        fig = plt.gcf()
        axes = fig.get_axes()
        
        # We want 5 even tick marks across the 224 pixels
        pixel_ticks = np.linspace(0, 224, 5)
        
        # Calculate exact Time text (0 seconds to the exact duration)
        time_vals = np.linspace(0, exact_duration, 5)
        time_labels = [f"{t:.1f}s" for t in time_vals]
        
        # Calculate exact Frequency text 
        freq_vals = np.linspace(nyquist_freq, 0, 5)
        freq_labels = [f"{int(f)}" for f in freq_vals]
        
        # Apply the math to the visual plots
        for ax in axes[:2]:
            ax.axis('on') # Turn borders back on
            
            # X-Axis (Time)
            ax.set_xticks(pixel_ticks)
            ax.set_xticklabels(time_labels, fontsize=8)
            ax.set_xlabel('Time', fontsize=10)
            
            # Y-Axis (Frequency)
            ax.set_yticks(pixel_ticks)
            ax.set_yticklabels(freq_labels, fontsize=8)
            ax.set_ylabel('Frequency (Hz)', fontsize=10)

    except FileNotFoundError:
        print(f"Could not find the original audio file at: {original_audio_path}")
        print("Axes will remain blank. Make sure the path logic above points to your .wav files!")


    # 4. Now finally display the finished window!
    print("Displaying Heatmap...")
    importance_map = np.sum(np.maximum(shap_array, 0), axis=-1)
    if importance_map.ndim == 3:
        importance_map = importance_map[0]
    # --- FIND THE EXACT PEAKS (TOP 5 FEATURES) ---
    # Assuming 'importance_map' is 2D (224, 224) array from the previous step

    N = 5 # Top features to find

    # 1. Flatten the map, sort it, and get the indices of the largest N values
    # (We use [::-1] to reverse it so the highest value is first)
    flat_indices = np.argsort(importance_map.flatten())[-N:][::-1]

    # 2. Convert those flattened indices back into 2D (y, x) pixel coordinates
    top_y_coords, top_x_coords = np.unravel_index(flat_indices, importance_map.shape)

    print(f"\n--- Top {N} Most Important Audio Features ---")

    # 3. Loop through the top points and translate each one to Time and Hz
    for i in range(N):
        x = top_x_coords[i]
        y = top_y_coords[i]
        
        # How "strong" the evidence was at this exact point
        shap_score = importance_map[y, x] 
        
        # Translate pixel 'x' to Time
        exact_time = (x / 224) * exact_duration
        
        # Translate pixel 'y' to Frequency (Hz)
        exact_hz = ((224 - y) / 224) * nyquist_freq
        
        print(f"Rank #{i+1}:")
        print(f"  - Time: {exact_time:.2f} seconds")
        print(f"  - Pitch: {int(exact_hz)} Hz")
        print(f"  - Impact Score: {shap_score:.5f}")
        print("-" * 30)

    file_name = f'{image_filename.split(".")[0]}'.replace("_spectrogram", "_shap_heatmap") + '.png'
    save_path = os.path.join(f"output/shap_heatmaps/{predicted_name}", file_name)
    os.makedirs(f"output/shap_heatmaps/{predicted_name}", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()