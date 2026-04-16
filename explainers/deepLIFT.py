import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import types
from PIL import Image
from torchvision import models, transforms
from torchvision.models.resnet import Bottleneck
from captum.attr import DeepLift
from captum.attr import visualization as viz

# ==========================================
# 1. ARCHITECTURE & MODEL SETUP
# ==========================================

def patch_resnet_for_deeplift(model):
    """
    Turns off inplace ReLUs and surgically replaces reused ReLUs in Bottlenecks.
    Required for Captum DeepLIFT to trace gradients without memory corruption.
    """
    # 1. Turn off global inplace operations
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

    # 2. Patch the Bottlenecks
    for name, module in model.named_modules():
        if isinstance(module, Bottleneck):
            module.relu1 = torch.nn.ReLU(inplace=False)
            module.relu2 = torch.nn.ReLU(inplace=False)
            module.relu3 = torch.nn.ReLU(inplace=False)

            def new_forward(self, x):
                identity = x
                out = self.relu1(self.bn1(self.conv1(x)))
                out = self.relu2(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                if self.downsample is not None:
                    identity = self.downsample(x)
                out += identity
                out = self.relu3(out)
                return out

            module.forward = types.MethodType(new_forward, module)
    return model

def load_binary_resnet(checkpoint_path, device):
    """Loads the ResNet50 model with 2 output classes."""
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = patch_resnet_for_deeplift(model)
    model.eval().to(device)
    return model

# ==========================================
# 2. DATA PROCESSING HELPER
# ==========================================

def get_audio_metadata(image_path, audio_dataset_dir):
    """Finds the original .wav file to extract true duration and frequency."""
    image_filename = os.path.basename(image_path)
    target_wav_name = image_filename.replace("_spectrogram.png", ".wav")
    
    for root, _, files in os.walk(audio_dataset_dir):
        if target_wav_name in files:
            wav_path = os.path.join(root, target_wav_name)
            y, sr = librosa.load(wav_path, sr=None)
            return librosa.get_duration(y=y, sr=sr), (sr / 2)
            
    raise FileNotFoundError(f"Could not find original audio for {target_wav_name}")

# ==========================================
# 3. CORE DEEPLIFT EXPLAINER
# ==========================================

def generate_deeplift_heatmap(image_path, model, audio_dir, class_names, device, show_plot=True):
    """Runs DeepLIFT and generates a high-resolution heatmap overlay."""
    print(f"Analyzing: {os.path.basename(image_path)}")
    
    # Prepare Image
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # Get Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = int(torch.argmax(outputs, dim=1))
        predicted_name = class_names[predicted_idx]

    # Run DeepLIFT
    dl = DeepLift(model)
    baseline = torch.zeros_like(input_tensor).to(device) # Baseline = pure silence
    attributions = dl.attribute(input_tensor, baselines=baseline, target=predicted_idx)
    
    # Format arrays for Captum visualization (H, W, C)
    attr_np = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    orig_np = np.transpose(input_tensor.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # Fetch Audio Physics for Axes
    duration, nyquist_freq = get_audio_metadata(image_path, audio_dir)

    # Plotting
    fig, axis = plt.subplots(1, 1, figsize=(8, 6))
    viz.visualize_image_attr(
        attr_np,
        orig_np,
        method="heat_map",          # <--- Change 1: Removes the background image
        sign="absolute_value",      # <--- Change 2: Standard for traditional saliency
        cmap="inferno",             # <--- Change 3: A classic high-contrast Saliency colormap
        show_colorbar=True,
        title=f"DeepLIFT Saliency Map (Predicted: {predicted_name})",
        plt_fig_axis=(fig, axis),
        use_pyplot=False
    )

    # Format Axes
    ticks = np.linspace(0, 224, 5)
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{t:.1f}s" for t in np.linspace(0, duration, 5)], fontsize=9)
    axis.set_xlabel("Time", fontsize=10)
    
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{int(f)}" for f in np.linspace(nyquist_freq, 0, 5)], fontsize=9)
    axis.set_ylabel("Frequency (Hz)", fontsize=10)

    # Save and Show
    os.makedirs("deeplift_output", exist_ok=True)
    filename = os.path.basename(image_path).replace("_spectrogram.png", "_deeplift.png")
    plt.savefig(os.path.join("deeplift_output", filename), dpi=300, bbox_inches="tight")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
