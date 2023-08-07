from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import hyperparameters
import matplotlib.pyplot as plt

def train(model, train_dataloader, valid_dataloader, num_epochs):
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters.lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    prev_train = 0
    prev_valid = 0

    plt.ion()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i, (images, labels) in enumerate(tqdm(train_dataloader)):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()
        
        train_loss /= i+1

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0

            for j, (images, labels) in enumerate(tqdm(valid_dataloader)):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumulate loss
                valid_loss += loss.item()

        valid_loss /= j+1

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

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
