import os
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import shap
import librosa

def shap_global_explain(target_folder_path, full_spectrogram_folder_for_background, original_audio_folder, model_checkpoint, class_name_input, show_plot=True):
# --- 1. SETUP ---
# Point this to the folder containing all the spectrograms
    image_paths = glob.glob(os.path.join(target_folder_path, "*.png"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(class_name_input)) # Dynamically set output layer size based on number of classes in the dataset

    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # --- SETUP DATA TRANSFORMS ---
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # These numbers are the standard PyTorch normalization math for ResNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # --- SETUP BACKGROUND DATA & EXPLAINER ---
    print("Loading background dataset for SHAP...")
    # Make sure this points to the main folder that contains the Cello, Violin, and Other subfolders
    full_dataset = datasets.ImageFolder(root=full_spectrogram_folder_for_background, transform=preprocess)
    background_loader = DataLoader(full_dataset, batch_size=50, shuffle=True)
    background_images, _ = next(iter(background_loader))
    background_images = background_images.to(device)

    print("Initializing GradientExplainer...")
    explainer = shap.GradientExplainer(model, background_images)
    # Create a blank 2D canvas to add all our SHAP values onto
    master_accumulator = np.zeros((224, 224))
    processed_count = 0

    print(f"Found {len(image_paths)} images. Starting Global SHAP calculation...")
    print("This may take a few minutes depending on the CPU/GPU...")

    # --- 2. LOOP THROUGH ALL IMAGES ---
    for path in image_paths:
        try:
            # Load and preprocess just one image at a time
            img = Image.open(path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            # Calculate SHAP for this single image
            shap_values = explainer.shap_values(img_tensor)
            class_name_target = os.path.basename(target_folder_path)
            class_index = class_name_input.index(class_name_target) # Dynamically find the index of the target class
            # Extract the SHAP array for the winning class
            # print(len(shap_values), shap_values[0].shape)
            shap_array = shap_values[0][:, :, :, class_index]

            # (3, 224, 224)
            # print(shap_array.shape)
            
            # Format the array properly (channels last)
            if shap_array.shape[0] == 3:
                shap_array = np.transpose(shap_array, (1, 2, 0))
            # print(shap_array.shape)
            # Compress colors to get the 2D map (positive evidence only)
            importance_map = np.sum(np.abs(shap_array), axis=-1)
            # print(importance_map.shape)
            # Add this image's evidence to the master pile
            master_accumulator += importance_map
            processed_count += 1

            # Print progress so we know it hasn't frozen
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} / {len(image_paths)}...")

        except Exception as e:
            print(f"Skipping {os.path.basename(path)} due to error: {e}")

    # --- 3. CALCULATE THE AVERAGE & DYNAMIC PHYSICS ---
    if processed_count > 0:
        print("\nCalculating final SHAP averages...")
        master_trend_map = master_accumulator / processed_count

        print("Checking original .wav files for exact Sample Rate and Average Duration...")
        total_duration = 0
        valid_audio_count = 0
        SAMPLE_RATE = 22050 # Fallback default
        
        # We loop through the image paths again, but this time we look for the .wav files
        for path in image_paths:
            filename = os.path.basename(path)
            target_wav_name = filename.replace("_spectrogram.png", ".wav")
            audio_path = None
            # Walk through the directory and all its subdirectories
            for root, dirs, files in os.walk(original_audio_folder):
                if target_wav_name in files:
                    audio_path = os.path.join(root, target_wav_name)
                    break  # Stop searching once we find the exact file

            # Safety check: Make sure we actually found it before proceeding
            if audio_path is None:
                print(f"Warning: Could not find {target_wav_name} in {original_audio_folder}")
                continue # Skip to the next image in your main loop
            
            try:
                # Load the audio file to check its exact physics
                y, sr = librosa.load(audio_path, sr=None)
                total_duration += librosa.get_duration(y=y, sr=sr)
                SAMPLE_RATE = sr  # Grab the sample rate directly from the file
                valid_audio_count += 1
            except Exception as e:
                print(f"Error loading audio file {audio_path}: {e}")
                pass # Skip if we can't find the .wav file

        # Calculate Average Duration
        if valid_audio_count > 0:
            AVG_DURATION = total_duration / valid_audio_count
            print(f"Success! Found Sample Rate: {SAMPLE_RATE} Hz | Average Duration: {AVG_DURATION:.2f}s")
        else:
            AVG_DURATION = 5.0 # Fallback if no .wav files were found
            print("Could not find .wav files. Using default fallback labels.")

        # --- 4. VISUALIZE THE MASTER HEATMAP ---
        class_name = os.path.basename(target_folder_path)
        print("Plotting the Global Master Heatmap...")
        plt.figure(figsize=(10, 8))
        plt.title(f"Master {class_name} Trend (Averaged over {processed_count} samples)", fontsize=14, fontweight='bold')

        plt.imshow(master_trend_map, cmap='coolwarm', interpolation='nearest')

        # Apply the dynamic physics to the axes
        NYQUIST_FREQ = SAMPLE_RATE / 2 
        pixel_ticks = np.linspace(0, 224, 5)
        
        # Generate labels using the Average Duration
        time_labels = [f"{t:.1f}s" for t in np.linspace(0, AVG_DURATION, 5)]
        freq_labels = [f"{int(f)}" for f in np.linspace(NYQUIST_FREQ, 0, 5)]
        
        plt.xticks(pixel_ticks, time_labels, fontsize=10)
        plt.yticks(pixel_ticks, freq_labels, fontsize=10)

        # Note that we label it "Average Time" so the user knows lengths varied
        plt.xlabel(f"Average Time (Seconds)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.colorbar(label="Average AI Importance Score")
        
        plt.tight_layout()
        file_name = f"Global_{class_name}_SHAP_Heatmap.png"
        save_path = os.path.join("output/shap_global", file_name)
        os.makedirs("output/shap_global", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
    else:
        print("No images were successfully processed. Check your folder path!")