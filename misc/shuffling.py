import os
import random
import shutil

def move_random_images(source_folder, dest_folder, num_images=10):
    # 1. Create the destination folder if it doesn't already exist
    os.makedirs(dest_folder, exist_ok=True)

    # 2. Get a list of all files in the source folder
    try:
        all_files = os.listdir(source_folder)
    except FileNotFoundError:
        print(f"Error: The source folder '{source_folder}' was not found.")
        return

    # 3. Filter the list to only include common image types
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
    image_files = [
        file for file in all_files 
        if file.lower().endswith(valid_extensions)
    ]

    # 4. Check if there are enough images to pick from
    if len(image_files) == 0:
        print("No images found in the source folder.")
        return
    elif len(image_files) < num_images:
        print(f"Warning: Found only {len(image_files)} images. Moving all of them.")
        selected_images = image_files
    else:
        # Randomly select the specified number of unique images
        selected_images = random.sample(image_files, num_images)

    # 5. Move (cut) the selected files to the destination
    moved_count = 0
    for image_name in selected_images:
        src_path = os.path.join(source_folder, image_name)
        dest_path = os.path.join(dest_folder, image_name)
        
        # shutil.move physically moves the file to the new location
        shutil.move(src_path, dest_path) 
        moved_count += 1
        print(f"Moved: {image_name}")

    print(f"\nSuccess! {moved_count} images moved to '{dest_folder}'.")


# --- How to use the script ---

# Replace these paths with the actual paths on your computer
SOURCE_DIRECTORY = ""
DESTINATION_DIRECTORY = ""

move_random_images(SOURCE_DIRECTORY, DESTINATION_DIRECTORY, num_images=10)