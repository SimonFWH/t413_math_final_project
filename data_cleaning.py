import os
import shutil

def move_empty_labels(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over the train, test, and valid folders
    for folder_name in ['train', 'test', 'valid']:
        if not os.path.exists(os.path.join(destination_folder, folder_name)):
            os.makedirs(os.path.join(destination_folder, folder_name))
        image_folder = os.path.join(source_folder, folder_name, 'images')
        label_folder = os.path.join(source_folder, folder_name, 'labels')

        # Iterate over the image folder
        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, filename.replace('.jpg', '.txt'))

            # Check if the label file is empty
            if os.path.exists(label_path) and os.path.getsize(label_path) == 0:
                # Move both the image and label files to the destination folder
                shutil.move(image_path, os.path.join(destination_folder, folder_name, 'images', filename))
                shutil.move(label_path, os.path.join(destination_folder, folder_name, 'labels', filename.replace('.jpg', '.txt')))
                print(f"Moved {filename} and its corresponding label file to {destination_folder}/{folder_name}")


# Example usage
source_folder = "data"
destination_folder = "empty"

move_empty_labels(source_folder, destination_folder)
