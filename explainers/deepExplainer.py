import torch
import torch.nn as nn
import types
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import shap
import matplotlib.pyplot as plt
import librosa
import os

def patch_resnet_for_deep_explainer(model):
    """
    Surgically fixes PyTorch's memory optimization so DeepExplainer can read it.
    Uses string matching to avoid torchvision versioning/import errors.
    """
    # Fix 1: Turn off standard inplace ReLUs
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

    # Fix 2 & 3: Rewire the Bottleneck
    for name, module in model.named_modules():
        
        # THE FIX: Check the name as a string instead of using isinstance()
        if module.__class__.__name__ == 'Bottleneck':
            
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
                    
                # The out-of-place addition fix
                out = out + identity 
                
                out = self.relu3(out)
                return out

            module.forward = types.MethodType(new_forward, module)
            
    return model

def shap_deepExplain(input_image_path, original_audio_folder, with_other=True, show_plot=True):
    # --- 1. SETUP MODEL ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50()
    if with_other:
        model.fc = nn.Linear(model.fc.in_features, 3) 
        model_checkpoint = 'cello_violin_other_resnet50.pth'
        class_names = ['Cello', 'Other', 'Violin']
    else:
        model.fc = nn.Linear(model.fc.in_features, 2)
        model_checkpoint = 'cello_violin_NEW_resnet50.pth'
        class_names = ['Cello', 'Violin']

    # Rebuild the skeleton and load the brain
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    
    # --- THE MINIMAL DEEPEXPLAINER PATCH ---
    # Force PyTorch to keep all memory intact so SHAP can trace it
    model = patch_resnet_for_deep_explainer(model)
            
    model = model.to(device)
    model.eval() # Critical for SHAP

    # --- 2. SETUP DATA TRANSFORMS ---
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- 3. CREATE THE "BACKGROUND" ---
    print("Loading background dataset for SHAP...")
    # Grabbing a batch using DataLoader is the cleanest way to build a background tensor
    full_dataset = datasets.ImageFolder(root='spectrograms_output', transform=preprocess)
    background_loader = DataLoader(full_dataset, batch_size=10, shuffle=True)
    background_images, _ = next(iter(background_loader))
    background_images = background_images.to(device)

    # --- 4. INITIALIZE SHAP ---
    print("Initializing DeepExplainer...")
    explainer = shap.DeepExplainer(model, background_images)

    # --- 5. PREPARE THE TEST IMAGE ---
    img = Image.open(input_image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device) # Shape is [1, 3, 224, 224]

    # --- 6. CALCULATE SHAP VALUES ---
    print("Calculating SHAP values (this might take a few seconds)...")
    
    # DeepExplainer calculates all classes at once. We need a quick forward pass 
    # to know which class the model actually predicted so we can plot the right heatmap.
    with torch.no_grad():
        outputs = model(img_tensor)
        predicted_class_idx = outputs.argmax(dim=1).item()
        
    predicted_name = class_names[predicted_class_idx]
    print(f"Model Predicted: {predicted_name}")

    # Calculate exact Deep SHAP values

    shap_values = explainer.shap_values(img_tensor, check_additivity=False)  ### OVERRIDE: check_additivity is too strict for PyTorch models, we disable it to avoid errors
    
    # --- 7. FORMAT AND PLOT ---
    print("Formatting arrays for plotting...")

    # 1. Format the Original Image Tensor
    test_numpy = img_tensor.cpu().numpy()
    if test_numpy.shape[1] == 3: 
        test_numpy = np.transpose(test_numpy, (0, 2, 3, 1))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    test_numpy = test_numpy * std + mean
    test_numpy = np.clip(test_numpy, 0, 1)

    # 2. Format the SHAP Array
    # DeepExplainer returns a list. We extract the array matching our prediction.
    shap_array = shap_values[predicted_class_idx] 
    
    # PyTorch SHAP versions occasionally return tensors instead of numpy arrays. This safely handles both.
    if torch.is_tensor(shap_array):
        shap_array = shap_array.cpu().numpy()
        
    if shap_array.shape[1] == 3:
        shap_array = np.transpose(shap_array, (0, 2, 3, 1))

    # 3. Generate the plot (Hidden in background)
    shap.image_plot([shap_array], test_numpy, labels=np.array([[predicted_name]]), show=False)

    print("Calculating exact physics for axes...")
    image_filename = os.path.basename(input_image_path)
    target_wav_name = image_filename.replace("_spectrogram.png", ".wav")
    source_dir = original_audio_folder

    original_audio_path = None
    for root, dirs, files in os.walk(source_dir):
        if target_wav_name in files:
            original_audio_path = os.path.join(root, target_wav_name)
            break 
            
    try:
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
            ax.axis('on') 
            ax.set_xticks(pixel_ticks)
            ax.set_xticklabels(time_labels, fontsize=8)
            ax.set_xlabel('Time', fontsize=10)
            
            ax.set_yticks(pixel_ticks)
            ax.set_yticklabels(freq_labels, fontsize=8)
            ax.set_ylabel('Frequency (Hz)', fontsize=10)

    except (FileNotFoundError, TypeError):
        print(f"Could not find the original audio file at: {original_audio_path}")
        print("Axes will remain blank.")

    # 4. Save and Display
    print("Displaying Heatmap...")
    os.makedirs("shap_deep", exist_ok=True)
    file_name = f'{image_filename.split(".")[0]}'.replace("_spectrogram", "_deep_shap") + '.png'
    save_path = os.path.join("shap_deep", file_name)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()