import os
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
from torchvision import models

# Grad-CAM specific imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import librosa

def gradcam_global_explain(target_folder_path, original_audio_folder, model_checkpoint, class_name_input, show_plot=False):
    # --- 1. SETUP ---
    image_paths = glob.glob(os.path.join(target_folder_path, "*.png"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(class_name_input))
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # Identify Target Layer for ResNet50
    target_layers = [model.layer4[-1]]

    # Setup Data Transforms (Same as SHAP)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create a blank 2D canvas to add all our Grad-CAM matrices onto
    master_accumulator = np.zeros((224, 224))
    processed_count = 0

    print(f"Found {len(image_paths)} images. Starting Global Grad-CAM calculation...")

    # Identify the target class index dynamically
    class_name_target = os.path.basename(os.path.normpath(target_folder_path))
    class_index = class_name_input.index(class_name_target)
    targets = [ClassifierOutputTarget(class_index)]

    # --- 2. LOOP THROUGH ALL IMAGES ---
    # We use the 'with' statement to ensure Grad-CAM hooks don't cause a GPU memory leak over large datasets
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for path in image_paths:
            try:
                # Load and preprocess
                img = Image.open(path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(device)

                # Calculate Grad-CAM for this single image
                # This natively outputs a (224, 224) matrix of floats between 0.0 and 1.0
                importance_map = cam(input_tensor=img_tensor, targets=targets)[0, :]

                # Add this image's evidence to the master pile
                master_accumulator += importance_map
                processed_count += 1

                # --- EXPORT INDIVIDUAL GRAD-CAM ARRAY ---
                base_filename = os.path.splitext(os.path.basename(path))[0]
                indiv_folder = os.path.join("output", "gradcam_individual", class_name_target)
                os.makedirs(indiv_folder, exist_ok=True)
                
                # Save the 224x224 importance map as a .npy file for the audio masker
                np.save(os.path.join(indiv_folder, f"{base_filename}_raw_gradcam.npy"), importance_map)

                # Print progress
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} / {len(image_paths)}...")

            except Exception as e:
                print(f"Skipping {os.path.basename(path)} due to error: {e}")

    # --- 3. CALCULATE THE AVERAGE & AUDIO PHYSICS ---
    if processed_count > 0:
        print("\nCalculating final Grad-CAM averages...")
        master_trend_map = master_accumulator / processed_count

        # --- EXPORT GLOBAL GRAD-CAM ARRAY ---
        os.makedirs("output/gradcam_global", exist_ok=True)
        np.save(f"output/gradcam_global/Global_{class_name_target}_raw_data.npy", master_trend_map)
        print(f"Saved raw global array to output/gradcam_global/Global_{class_name_target}_raw_data.npy")

        print("Checking original .wav files for exact Sample Rate and Average Duration...")
        total_duration = 0
        valid_audio_count = 0
        SAMPLE_RATE = 22050 # Fallback default
        
        for path in image_paths:
            filename = os.path.basename(path)
            target_wav_name = filename.replace("_spectrogram.png", ".wav")
            audio_path = None
            
            for root, dirs, files in os.walk(original_audio_folder):
                if target_wav_name in files:
                    audio_path = os.path.join(root, target_wav_name)
                    break 

            if audio_path is None:
                continue 
            
            try:
                y, sr = librosa.load(audio_path, sr=None)
                total_duration += librosa.get_duration(y=y, sr=sr)
                SAMPLE_RATE = sr 
                valid_audio_count += 1
            except Exception as e:
                pass 

        if valid_audio_count > 0:
            AVG_DURATION = total_duration / valid_audio_count
            print(f"Success! Found Sample Rate: {SAMPLE_RATE} Hz | Average Duration: {AVG_DURATION:.2f}s")
        else:
            AVG_DURATION = 5.0 
            print("Could not find .wav files. Using default fallback labels.")

        # --- 4. VISUALIZE THE MASTER HEATMAP ---
        print("Plotting the Global Master Heatmap...")
        plt.figure(figsize=(10, 8))
        plt.title(f"Master {class_name_target} Trend (Averaged over {processed_count} samples)", fontsize=14, fontweight='bold')

        # Using 'jet' colormap as it is the standard for Grad-CAM
        plt.imshow(master_trend_map, cmap='jet', interpolation='bilinear')

        NYQUIST_FREQ = SAMPLE_RATE / 2 
        pixel_ticks = np.linspace(0, 224, 5)
        
        time_labels = [f"{t:.1f}s" for t in np.linspace(0, AVG_DURATION, 5)]
        freq_labels = [f"{int(f)}" for f in np.linspace(NYQUIST_FREQ, 0, 5)]
        
        plt.xticks(pixel_ticks, time_labels, fontsize=10)
        plt.yticks(pixel_ticks, freq_labels, fontsize=10)

        plt.xlabel("Average Time (Seconds)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.colorbar(label="Average Class Activation Score")
        
        plt.tight_layout()
        file_name = f"Global_{class_name_target}_GradCAM_Heatmap.png"
        save_path = os.path.join("output/gradcam_global", file_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    else:
        print("No images were successfully processed. Check your folder path!")