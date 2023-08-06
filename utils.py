import os
import numpy as np
import torch
import hyperparameters
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset, DataLoader

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = os.listdir(folder_path+"/images")
        self.label_paths = os.listdir(folder_path+"/labels")
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.folder_path, "images", self.image_paths[index])
        label_path = os.path.join(self.folder_path, "labels", self.label_paths[index])
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((256, 256))
        except:
            print(f"Cannot open {image_path}")
        if self.transform:
            image = self.transform(image)

        with open(label_path) as f:
            label = np.array(f.readline().split()[1:]).astype('float32')
            label = torch.Tensor(label)
        return image, label

    def __len__(self):
        return len(self.image_paths)


def load_images_from_folder(folder_path):
    transform = Compose([
                        Resize((256, 256)),   # Resize the image to a fixed size
                        ToTensor()            # Convert the image to a tensor
                        ])
    dataset = ImageFolderDataset(folder_path, transform=transform)
    batch_size = hyperparameters.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dataloader

def save_model(model):
    model_folder = "model"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    # Save the model
    torch.save(model.state_dict(), f"{model_folder}/model.pth")