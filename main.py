import utils
import model
import train
import hyperparameters
import torch

if __name__ == "__main__":
    train_path = "data/train"
    valid_path = "data/valid"
    train_dataloader = utils.load_images_from_folder(train_path)
    valid_dataloader = utils.load_images_from_folder(valid_path)
    model = model.Model()
    train.train(model, train_dataloader,valid_dataloader, hyperparameters.num_epochs)
    utils.save_model(model)
    model.load_state_dict(torch.load("model/model.pth")