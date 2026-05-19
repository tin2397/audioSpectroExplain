"""
Main Execution Script for Audio Spectrogram SHAP Explanations
-------------------------------------------------------------
This script runs various model explainability techniques (like SHAP and GradCAM) 
on spectrograms generated from audio files. 

NOTES & SETUP:
1. Directory Structure: 
   Ensure your datasets and models are placed in the respective relative 
   directories (e.g., './dataset', './trained_models', './output').
2. Class Count:
   The number of classes is determined by the number of subfolders in your dataset.
   (e.g., Cello, Violin, and Other = 3 classes. Cello and Violin = 2 classes).
   Ensure the loaded model checkpoint matches this class count.
3. Outputs:
   Explanation plots are saved in the './output/[explanation_type]' folder, named 
   after the original spectrogram files.
4. Known Limitations:
   DeepLift and shap.DeepExplainer currently have compatibility issues with ResNet50.

AVAILABLE MODELS:
- cello_violin_NEW_resnet50: 2-class model (Cello vs. Violin).
- cello_violin_other_resnet50: 3-class model (Cello, Violin, Other).
- penguin_resnet50: 2-class model for the Penguin dataset (13B3-1 vs 14B19-1).
"""

import os
import numpy as np
# --- Imports (Uncomment specific explainers or utilities as needed) ---
# from model.predict import predict_instrument
from explainers.SHAP import shap_explain
from explainers.globalSHAP import shap_global_explain
# from explainers.SHAPparti import shap_explain_partition
# from explainers.gradCAM import gradCAM_explain
# from misc.audio_to_spectrogram import generate_spectrogram
from misc.maskingAudio import apply_shap_mask_to_audio

from explainers.globalGradCAM import gradcam_global_explain

# ==========================================
# CONFIGURATION
# ==========================================

# Use relative paths so this works on any machine (GitHub friendly)
PROJECT_ROOT = "."
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
SPECTROGRAM_DIR = os.path.join(PROJECT_ROOT, "spectrograms_output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")

# --- Active Configuration: Cello/Violin ---
MODEL_CHECKPOINT = os.path.join(MODELS_DIR, "cello_violin_NEW_resnet50.pth")

# Specific test inputs
INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "blind_test", "Cello", "Vc-ord-F3-mf-3c-T11d_spectrogram.png")
ORIGINAL_AUDIO_FOLDER = os.path.join(DATASET_DIR, "TinySOL", "TinySOL2020") # POINT TO THE MAIN FOLDER THAT CONTAINS ALL SUBFOLDERS THAT WE WANT TO SEARCH THROUGH TO FIND THE ORIGINAL AUDIO FILES (e.g., "TinySOL" folder that contains all the instrument subfolders and .wav files)
GLOBAL_TARGET_FOLDER = os.path.join(SPECTROGRAM_DIR, "Instrument", "Cello") # POINT TO THE FOLDER CONTAINING THE SPECTROGRAMS YOU WANT TO EXPLAIN GLOBALLY (e.g., all Violin spectrograms)
BACKGROUND_FOLDER = os.path.join(SPECTROGRAM_DIR, "Instrument") # POINT TO THE MAIN FOLDER THAT CONTAINS ALL SUBFOLDERS THAT WE WANT TO USE AS BACKGROUND 

CLASS_NAMES = sorted([
    folder_name for folder_name in os.listdir(BACKGROUND_FOLDER) 
    if os.path.isdir(os.path.join(BACKGROUND_FOLDER, folder_name))
])
# --- Inactive Configuration: Penguin (Example) ---
# INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "blind_test", "Penguin", "13B3-1", "13B3-1_exhale-i16_83_spectrogram.png")
# ORIGINAL_AUDIO_FOLDER = os.path.join(DATASET_DIR, "Pen")
# MODEL_CHECKPOINT = os.path.join(MODELS_DIR, "penguin_resnet50.pth")
# CLASS_NAMES = ['13B3-1', '14B19-1']


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

# def run_all_explainers_folder(folder_path, original_audio_folder, model_checkpoint, class_names, bg_folder, show_plot=False):
#     """Generates all explainers for a folder of spectrograms."""
#     error_count = 0
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith("_spectrogram.png"):
#                 image_path = os.path.join(root, file)
#                 try:
#                     # Note: Uncomment imports at the top to use these
#                     shap_explain(image_path, bg_folder, original_audio_folder, model_checkpoint, class_names, show_plot=show_plot)
#                     # shap_explain_partition(image_path, original_audio_folder, model_checkpoint, class_names, show_plot=show_plot)
#                     # gradCAM_explain(image_path, original_audio_folder, model_checkpoint, class_names, show_plot=show_plot)
#                 except Exception as e:
#                     print(f"Error processing {image_path}: {e}")
#                     error_count += 1
#     print(f"Finished processing folder. Total errors: {error_count}")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # 1. GLOBAL SHAP EXPLANATION
    # ---------------------------------------------------------
    # Note: Global SHAP usually takes ~6.5GB VRAM on a full folder (e.g., 274 images).
    # Takes ~10 minutes on an NVIDIA 3070TI. CPU execution is significantly slower.
    
   # print("Starting Global SHAP Explanation...")
   # shap_global_explain(
   #    target_folder_path=GLOBAL_TARGET_FOLDER,
   #    full_spectrogram_folder_for_background=BACKGROUND_FOLDER,
   #    original_audio_folder=ORIGINAL_AUDIO_FOLDER,
   #    model_checkpoint=MODEL_CHECKPOINT,
   #    class_name_input=CLASS_NAMES,
   #    show_plot=False
   # )

    # ---------------------------------------------------------
    # 2. SINGLE INFERENCE / PREDICTION (Commented Out)
    # ---------------------------------------------------------
    # label, score = predict_instrument(INPUT_IMAGE_PATH)
    # print(f"Prediction: {label} ({score*100:.2f}%)")

    # ---------------------------------------------------------
    # 3. SPECTROGRAM GENERATION (Commented Out)
    # ---------------------------------------------------------
    # Example: Generate for a single file
    # generate_spectrogram("sample.wav", output_path="./output_folder", human_readable=False)
    
    # Example: Batch generate for an entire folder
    # audio_target_dir = os.path.join(DATASET_DIR, "Pen", "14B19-1_clipped")
    # spectro_output_dir = os.path.join(SPECTROGRAM_DIR, "Penguin", "14B19-1_clipped")
    # for root, dirs, files in os.walk(audio_target_dir):
    #     for file in files:
    #         if file.endswith(".wav"):
    #             audio_path = os.path.join(root, file)
    #             generate_spectrogram(audio_path, output_path=spectro_output_dir, human_readable=False)



    # ---------------------------------------------------------
    # 4. SINGLE IMAGE EXPLAINERS (Commented Out)
    # ---------------------------------------------------------
   #  shap_explain(
   #      input_image_path=INPUT_IMAGE_PATH, 
   #      full_spectrogram_folder_for_background=BACKGROUND_FOLDER, 
   #      original_audio_folder=ORIGINAL_AUDIO_FOLDER, 
   #      model_checkpoint=MODEL_CHECKPOINT, 
   #      class_names=CLASS_NAMES, 
   #      show_plot=True)
    
    # shap_explain_partition(
    #     input_image_path=INPUT_IMAGE_PATH, 
    #     full_spectrogram_folder_for_background=BACKGROUND_FOLDER, 
    #     original_audio_folder=ORIGINAL_AUDIO_FOLDER, 
    #     model_checkpoint=MODEL_CHECKPOINT, 
    #     class_names=CLASS_NAMES, 
    #     show_plot=False)
    
    # gradCAM_explain(
    #     input_image_path=INPUT_IMAGE_PATH, 
    #     original_audio_folder=ORIGINAL_AUDIO_FOLDER, 
    #     model_checkpoint=MODEL_CHECKPOINT, 
    #     class_names=CLASS_NAMES, 
    #     show_plot=False)



   # ---------------------------------------------------------
   # 5. GLOBAL GRAD-CAM EXPLANATION
   # ---------------------------------------------------------
   print("Starting Global Grad-CAM Explanation...")
   gradcam_global_explain(
      target_folder_path=GLOBAL_TARGET_FOLDER,
      original_audio_folder=ORIGINAL_AUDIO_FOLDER,
      model_checkpoint=MODEL_CHECKPOINT,
      class_name_input=CLASS_NAMES,
      show_plot=False
   )

   """# ----------------------MASKING AUDIO WITH SHAP VALUES (Commented Out) ---------------------------------------------------------"""
   # SHAP_INDIVIDUAL_DIR = os.path.join(PROJECT_ROOT, "output", "shap_individual")
   # OUTPUT_MASKED_AUDIO_DIR = os.path.join(PROJECT_ROOT, "output", "masked_audio")
   # os.makedirs(OUTPUT_MASKED_AUDIO_DIR, exist_ok=True)


   # target_class = "14B19-1" # CHANGE THIS TO THE TARGET CLASS YOU WANT TO USE (e.g., "Cello", "Violin", "13B3-1", "14B19-1")

   # # Update this 
   # shap_filename = "14B19-1_exhale-i1_54_spectrogram_raw_shap.npy"

   # file_name = shap_filename.replace("_spectrogram_raw_shap.npy", "")
   # os.makedirs(os.path.join(OUTPUT_MASKED_AUDIO_DIR, target_class, file_name), exist_ok=True)
   # os.makedirs(os.path.join(OUTPUT_MASKED_AUDIO_DIR, target_class, file_name, "Plot"), exist_ok=True)

   # #shap_file_path = os.path.join(SHAP_INDIVIDUAL_DIR, target_class, shap_filename)
   # shap_file_path = r"C:\Users\tin23\OneDrive\Desktop\HONS WORKING\audioSpectroExplain\output\shap_global\Global_14B19-1_raw_data.npy"

   # base_audio_name = shap_filename.replace("_spectrogram_raw_shap.npy", ".wav")

   # # Search the TinySOL directory to find where the .wav 
   # audio_file_path = None
   # for root, dirs, files in os.walk(ORIGINAL_AUDIO_FOLDER):
   #    if base_audio_name in files:
   #       audio_file_path = os.path.join(root, base_audio_name)
   #       break # Stop searching once found

   # # Execute the masking if both files exist
   # print(f"Found Audio: {audio_file_path}")
   # print(f"Found SHAP Map: {shap_file_path}")
   
   # loaded_shap_array = np.load(shap_file_path)
   # output_wav = os.path.join(OUTPUT_MASKED_AUDIO_DIR, target_class, file_name,f"masked_{base_audio_name}")
   
   # apply_shap_mask_to_audio(
   #    audio_file=audio_file_path,
   #    shap_importance_map=loaded_shap_array,
   #    output_wav_path=output_wav,
   #    output_image_path=os.path.join(OUTPUT_MASKED_AUDIO_DIR, target_class, file_name, "Plot", f"plot_{base_audio_name.replace('.wav', '.png')}")
   # )


