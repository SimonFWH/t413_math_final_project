import os
import numpy as np
import torch
from torchvision.transforms import ToTensor, Resize, Compose
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def infer(model, test_path):
    image_folder = os.path.join(test_path, "images")
    label_folder = os.path.join(test_path, "labels")
    model.eval()

    # Get a list of image files
    image_files = os.listdir(image_folder)

    # Create a figure and axis for the grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    start_index = 0
    while start_index < len(image_files):
        # Clear the previous grid
        for ax in axes.ravel():
            ax.clear()
        for i, image_file in enumerate(image_files[start_index:start_index+9]):
            # Load the image
            image_path = os.path.join(image_folder, image_file)
            label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + ".txt")
            image = Image.open(image_path).convert("RGB")
            # Preprocess the image
            transform = Compose([
                Resize((256, 256)),   # Resize the image to a fixed size
                ToTensor()            # Convert the image to a tensor
            ])
            input_image = transform(image).unsqueeze(0)
            plot_image = input_image.squeeze(0).permute(1, 2, 0).numpy()
            # Make predictions with the model
            with torch.no_grad():
                output = model(input_image)
            # Extract predicted bounding box
            pred_box = output.squeeze().tolist()

            with open(label_path, "r") as f:
                label = np.array(f.readline().split()[1:]).astype('float32')
                label = torch.Tensor(label)

            # Display the image with bounding boxes
            ax = axes[i // 3, i % 3]
            ax.imshow(plot_image)

            # Scale the label coordinates to match the displayed image size
            image_width, image_height = 256, 256
            x_center, y_center, width, height = label
            x_min = (x_center - width / 2) * image_width
            y_min = (y_center - height / 2) * image_height
            box_width = width * image_width
            box_height = height * image_height
            # Draw label bounding box in green
            label_rect = patches.Rectangle((x_min, y_min), box_width, box_height,
                                        linewidth=2, edgecolor="g", facecolor="none")
            ax.add_patch(label_rect)

            # Scale the predicted bounding box coordinates
            x_center, y_center, width, height = pred_box
            x_min = (x_center - width / 2) * image_width
            y_min = (y_center - height / 2) * image_height
            box_width = width * image_width
            box_height = height * image_height

            # Draw predicted bounding box in red
            pred_rect = patches.Rectangle((x_min, y_min), box_width, box_height,
                                        linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(pred_rect)

            # Set axis labels and remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Image {}".format(i+1))

        # Adjust spacing and display the grid of images
        plt.tight_layout()
        plt.draw()
        plt.waitforbuttonpress()
        # Increment the start index
        start_index += 9
    # Close the figure
    plt.close(fig)    
