import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from PIL import Image
from torchvision import models, transforms
from skimage.feature import peak_local_max

def shap_explain_partition(input_image_path, original_audio_folder, model_checkpoint, class_names, show_plot=True):
    # --- 1. SETUP MODEL ---
    class_size = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, class_size)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval().to(device)

    # PartitionExplainer sends batches of images as NumPy arrays.
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def predict_func(images):
        img_tensors = torch.stack([preprocess(img) for img in images]).to(device)

        with torch.no_grad():
            logits = model(img_tensors)
            # return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            return logits.cpu().numpy()

    # --- 2. INITIALIZE PARTITION EXPLAINER ---
    masker = shap.maskers.Image("blur(16, 16)", (224, 224, 3))
    explainer = shap.PartitionExplainer(predict_func, masker)

    # --- 3. PREPARE THE TEST IMAGE ---
    img = Image.open(input_image_path).convert("RGB")
    img_np = np.array(img.resize((224, 224)))

    predictions = predict_func(np.expand_dims(img_np, axis=0))
    predicted_class_idx = int(np.argmax(predictions[0]))
    predicted_name = class_names[predicted_class_idx]

    # --- 4. CALCULATE SHAP VALUES ---
    shap_values = explainer(
        np.expand_dims(img_np, axis=0),
        max_evals=3000,
        batch_size=64,
        outputs=shap.Explanation.argsort.flip[:1],
    )

    # --- 5. PLOT ---
    shap.image_plot(shap_values, labels=np.array([[predicted_name]]),show=False)

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

    fig = plt.gcf()
    axes = fig.get_axes()
    pixel_ticks = np.linspace(0, 224, 5)

    time_vals = np.linspace(0, exact_duration, 5)
    time_labels = [f"{t:.1f}s" for t in time_vals]

    freq_vals = np.linspace(nyquist_freq, 0, 5)
    freq_labels = [f"{int(f)}" for f in freq_vals]

    for ax in axes[:2]:
        ax.axis("on")
        ax.set_xticks(pixel_ticks)
        ax.set_xticklabels(time_labels, fontsize=8)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_yticks(pixel_ticks)
        ax.set_yticklabels(freq_labels, fontsize=8)
        ax.set_ylabel("Frequency (Hz)", fontsize=10)
        ax.set_title(predicted_name, fontsize=10)
    # --- 6. EXTRACT TOP 5 FEATURES ---
    print("\nExtracting exact physics for top features...")

    # 1. Unpack the Explanation Object
    raw_shap_array = shap_values.values[0, ..., 0] 
    importance_map = np.sum(np.maximum(raw_shap_array, 0), axis=-1)

    print("\n--- Top 5 Most Important Audio Features ---")

    # 2. Let skimage handle the spatial filtering automatically
    coordinates = peak_local_max(importance_map, min_distance=15, num_peaks=5)

    # 3. Print the results
    for rank, (y, x) in enumerate(coordinates):
        shap_score = importance_map[y, x]
        
        # Translate pixels to real-world physics
        exact_time = (x / 224) * exact_duration
        exact_hz = ((224 - y) / 224) * nyquist_freq
        
        print(f"Rank #{rank+1}:")
        print(f"  - Time: {exact_time:.2f} seconds")
        print(f"  - Pitch: {int(exact_hz)} Hz")
        print(f"  - Impact Score: {shap_score:.5f}")
        print("-" * 30)

    os.makedirs(f"output/shap_partition/{predicted_name}", exist_ok=True)
    file_name = image_filename.replace("_spectrogram.png", "_partition.png")
    save_path = os.path.join(f"output/shap_partition/{predicted_name}", file_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return save_path, predicted_name
