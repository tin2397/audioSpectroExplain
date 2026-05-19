import librosa
import numpy as np
import soundfile as sf
# https://www.geeksforgeeks.org/python/image-resizing-using-opencv-python/
import cv2  
import matplotlib.pyplot as plt
import librosa.display
import os

def apply_shap_mask_to_audio(audio_file, shap_importance_map, output_wav_path, output_image_path=None):
    if isinstance(shap_importance_map, str):
        shap_importance_map = np.load(shap_importance_map)
    shap_importance_map = np.flipud(shap_importance_map)  # Flip vertically to match spectrogram orientation
    # 1. Load original audio and get STFT
    y, sr = librosa.load(audio_file, sr=None)
    D = librosa.stft(y)
    
    original_height, original_width = D.shape
    
    shap_importance_map = shap_importance_map.astype(np.float32)
    print("SHAP importance map loaded.")
    # 2. Stretch the 224x224 SHAP map to high-res STFT dimensions
    stretched_mask = cv2.resize(shap_importance_map, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    
    # 3. Create the Binary Filter (Thresholding)
    threshold_value = np.percentile(stretched_mask, 98)
    binary_mask = np.where(stretched_mask > threshold_value, 1.0, 0.0)
    # binary_mask = np.where(stretched_mask > 0, 1.0, 0.0)

    # Compute the original dB spectrogram so we know its exact color range
    original_S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    original_vmin = original_S_db.min()
    original_vmax = original_S_db.max() # This will be 0.0

    # 4. Apply Mask to the complex data
    masked_D = D * binary_mask
    original_peak_amplitude = np.max(np.abs(D))
    # 5. Convert to dB for plotting
    # We take the magnitude of the masked data and convert to decibels
    masked_S_db = librosa.amplitude_to_db(np.abs(masked_D), ref=original_peak_amplitude)
    
    # Find the actual maximum dB left in the masked audio
    masked_max = masked_S_db.max()
    # --- PLOTTING LOGIC ---
    plt.figure(figsize=(15, 5))

    # Plot 1: The Masked Spectrogram (The "Filtered" Sound)
    # plt.subplot(Rows, Columns, Index). Here we want 1 row, 2 columns, and this is the first plot (index=1)
    plt.subplot(1, 2, 1)
    librosa.display.specshow(
        masked_S_db, 
        sr=sr, 
        x_axis='time', 
        y_axis='linear',
        vmin=original_vmin,
        vmax=original_vmax)
    cbar = plt.colorbar(format='%+2.0f dB')

    cbar.ax.set_ylim(original_vmin, masked_max)
    plt.title("Spectrogram AI Focus Only")

    # Plot 2: The Importance Map (The Stretched SHAP Mask)
    plt.subplot(1, 2, 2)
    # Using 'RdBu_r' colormap to maintain SHAP's Red (Positive) / Blue (Negative) look
    plt.imshow(stretched_mask, aspect='auto', origin='lower', cmap='coolwarm', 
               extent=[0, len(y)/sr, 0, sr/2])
    plt.colorbar(label='SHAP Importance Score')
    plt.title("SHAP Importance Map")

    plt.tight_layout()
    reconstructed_audio = librosa.istft(masked_D)
    sf.write(output_wav_path, reconstructed_audio, sr)
    
    print(f"Success! Audio saved to: {output_wav_path}")
    if output_image_path is not None:
        # Automatically create the destination folder if it doesn't exist
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # Save at 300 DPI for high-quality academic/thesis rendering
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"Success! Plot image saved to: {output_image_path}")
    plt.show()

    # 6. Save the new audio
