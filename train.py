from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import hyperparameters
import matplotlib.pyplot as plt
import utils

def train(model, train_dataloader, valid_dataloader, num_epochs):
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters.lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    prev_train = 0
    prev_valid = 0

    best_valid_loss = float('inf')
    current_patience = 0

    plt.ion()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        valid_iou = 0.0

        for i, (images, labels) in enumerate(tqdm(train_dataloader)):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate IoU per sample and accumulate the sum
            for j in range(len(outputs)):
                iou = utils.calculate_iou(labels[j].tolist(), outputs[j].tolist())
                train_iou += iou

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()
        
        train_loss /= i+1
        train_iou /= len(train_dataloader.dataset)

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0

            for j, (images, labels) in enumerate(tqdm(valid_dataloader)):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                # Calculate IoU per sample and accumulate the sum
                for j in range(len(outputs)):
                    iou = utils.calculate_iou(labels[j].tolist(), outputs[j].tolist())
                    valid_iou += iou

                # Accumulate loss
                valid_loss += loss.item()

        valid_loss /= j+1
        valid_iou /= len(valid_dataloader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss}, Validation Loss: {valid_loss}, Train IoU: {train_iou:.4f}, Validation IoU: {valid_iou:.4f}')

        # Plot the training and validation loss for each epoch
        if epoch != 0:
            plt.plot([epoch - 1, epoch],
                    [prev_train, train_loss],
                    color="green",
                    label="Train Loss" if epoch==1 else "")
            plt.plot([epoch - 1, epoch],
                    [prev_valid, valid_loss],
                    color="red",
                    label="Validation Loss" if epoch==1 else "")
            plt.pause(0.1)
            plt.legend()
            plt.show()
        
        prev_train = train_loss
        prev_valid = valid_loss
