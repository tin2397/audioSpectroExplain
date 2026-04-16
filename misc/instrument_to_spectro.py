import pandas as pd
from misc.audio_to_spectrogram import generate_spectrogram
import os

# Load metadata
metadata = pd.read_csv("TinySOL/TinySOL_metadata.csv")
# Process each audio file
target_instruments = ['Violin','Cello']
counts = {'Violin': 0, 'Cello': 0, 'Other': 0}

print(f"Starting processing for instruments: {', '.join(target_instruments)}")
# 3. Process ALL audio files 
for idx, row in metadata.iterrows():
    actual_instrument = row['Instrument (in full)']
    
    # --- THE SORTING LOGIC ---
    # If it's a target instrument, keep its name. Otherwise, label it "Other"
    if actual_instrument in target_instruments:
        folder_label = actual_instrument
    else:
        folder_label = "Other"
    
    audio_path = 'TinySOL/TinySOL2020/' + row['Path']
    
    # Put the file in the correct sub-folder based on our sorting logic
    output_dir = f"spectrograms_output/{folder_label}"
    
    # Ensure the output directory actually exists before saving
    os.makedirs(output_dir, exist_ok=True)

    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"WARNING: Audio file not found: {audio_path}. Skipping.")
        continue 
    
    # Generate the spectrogram
    generate_spectrogram(audio_path, output_path=output_dir)
    
    # Increment the correct counter
    counts[folder_label] += 1

# 4. Print Summary Report
print("\n" + "="*30)
print("PROCESSING COMPLETE")
print("="*30)
print(f"Violin files processed: {counts['Violin']}")
print(f"Cello files processed:  {counts['Cello']}")
print(f"Other files processed:  {counts['Other']}")
print(f"Total files:            {sum(counts.values())}")
print("="*30)