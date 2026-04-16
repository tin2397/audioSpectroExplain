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

# --- Imports (Uncomment specific explainers or utilities as needed) ---
# from model.predict import predict_instrument
from explainers.SHAP import shap_explain
from explainers.globalSHAP import shap_global_explain
# from explainers.SHAPparti import shap_explain_partition
# from explainers.gradCAM import gradCAM_explain
# from misc.audio_to_spectrogram import generate_spectrogram


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
CLASS_NAMES = ['Cello', 'Violin']  # MUST match the order of subfolders / model output layer

# Specific test inputs
INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "blind_test", "Violin", "Vn-ord-D#6-ff-1c-N_spectrogram.png")
ORIGINAL_AUDIO_FOLDER = os.path.join(DATASET_DIR, "TinySOL", "TinySOL2020")
GLOBAL_TARGET_FOLDER = os.path.join(SPECTROGRAM_DIR, "Instrument", "Violin")
BACKGROUND_FOLDER = os.path.join(SPECTROGRAM_DIR, "Instrument") # POINT TO THE MAIN FOLDER THAT CONTAINS ALL SUBFOLDERS THAT WE WANT TO USE AS BACKGROUND 

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
    
   #  print("Starting Global SHAP Explanation...")
   #  shap_global_explain(
   #      target_folder_path=GLOBAL_TARGET_FOLDER,
   #      full_spectrogram_folder_for_background=BACKGROUND_FOLDER,
   #      original_audio_folder=ORIGINAL_AUDIO_FOLDER,
   #      model_checkpoint=MODEL_CHECKPOINT,
   #      class_name_input=CLASS_NAMES,
   #      show_plot=True
   #  )

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
    shap_explain(
        input_image_path=INPUT_IMAGE_PATH, 
        full_spectrogram_folder_for_background=BACKGROUND_FOLDER, 
        original_audio_folder=ORIGINAL_AUDIO_FOLDER, 
        model_checkpoint=MODEL_CHECKPOINT, 
        class_names=CLASS_NAMES, 
        show_plot=True)
    
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