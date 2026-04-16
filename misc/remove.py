import os
import random

# Path to your 'Other' folder
other_folder = "spectrograms_output/Other"

# How many files do you want to keep? (Match your Violin/Cello count)
TARGET_COUNT = 300 

# Get all the files in the folder
all_files = os.listdir(other_folder)

# Check if we actually need to delete anything
if len(all_files) > TARGET_COUNT:
    print(f"Found {len(all_files)} files. Trimming down to {TARGET_COUNT}...")
    
    # Randomly shuffle the list so we get a good mix of flutes, drums, etc.
    random.shuffle(all_files)
    
    # Keep the first 300, select the rest for deletion
    files_to_delete = all_files[TARGET_COUNT:]
    
    # Delete the excess files
    for file_name in files_to_delete:
        file_path = os.path.join(other_folder, file_name)
        os.remove(file_path)
        
    print("Done! The 'Other' folder is now perfectly balanced.")
else:
    print(f"Folder only has {len(all_files)} files. No need to delete!")