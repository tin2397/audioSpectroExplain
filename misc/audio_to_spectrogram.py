# https://zenodo.org/records/3685331
# https://www.kaggle.com/datasets/thedevastator/tinysol-isolated-musical-notes-from-14-musical-i
# https://www.kaggle.com/datasets/abdulvahap/music-instrunment-sounds-for-classification 

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def generate_spectrogram(audio_file, output_path=None, human_readable=False):
    y, sr = librosa.load(audio_file, sr=None)

    D = librosa.stft(y)
    S = np.abs(D)
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))

    # --- THE TOGGLE SWITCH ---
    if human_readable:
        # 1. The version for HUMAN EYES, with axes and colorbar
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {os.path.basename(audio_file)}')
    else:
        # 2. The "pure" version for the ResNet
        librosa.display.specshow(S_db, sr=sr)
        plt.axis('off')

    # --- SAVING LOGIC ---
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.basename(audio_file)
        name_only = os.path.splitext(base_name)[0]
        file_name = f'{name_only}_spectrogram.png'
        save_file = os.path.join(output_path, file_name)
        
        if human_readable:
            # Save normally with all the borders and text
            plt.savefig(save_file)
        else:
            # Save cleanly with no white borders
            plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
            
        plt.close()
        
    # --- DISPLAY LOGIC ---
    else:
        if not human_readable:
            # Strip borders for the pop-up window
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
        plt.show()
        plt.close()