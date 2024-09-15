import torch
import torch.nn as nn
import torch.optim as optim
import onnx
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np

# Define the MLPModel class
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
    
    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_loader, val_loader, device, mse_threshold=0.01, max_epochs=1000):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    max_epochs = max_epochs
    mse_threshold = mse_threshold
    best_val_loss = float('inf')

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')


        # Append current losses to the lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch == 0:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            train_line, = ax.plot(train_losses, label='Training Loss')
            val_line, = ax.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss over Epochs')
        else:
            train_line.set_data(range(len(train_losses)), train_losses)
            val_line.set_data(range(len(val_losses)), val_losses)
            ax.relim()
            ax.autoscale_view()

        plt.draw()
        plt.pause(0.001)  # Pause to update the plot


        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('Validation loss improved')
            

        if val_loss <= mse_threshold:
            print('Desired MSE threshold reached. Stopping training.')
            torch.save(model.state_dict(), 'best_model.pth')
            break

if __name__ == '__main__':

    # Generate synthetic data
    # X = np.random.rand(1000, 5).astype(np.float32)
    # y = np.random.rand(1000, 4).astype(np.float32)
    # X = torch.from_numpy(X)
    # y = torch.from_numpy(y)

    # Load the training data
    # train_inputs = np.load('train_inputs.npy')

    # x = train_inputs[:, :-4]
    # y = train_inputs[:, -4:]

    data_file_agentic = 'eps_100_runs_100.pt'
    data_file_random = 'random_eps_100_runs_100.pt'

    try:
        data_dict_1 = torch.load(data_file_agentic)
        data_dict_2 = torch.load(data_file_random)
        print('Data loaded successfully from .pt files.')
    except Exception as e:
        print(f'Error loading data: {e}')
        exit(1)

    # Initialize lists to store inputs and targets
    inputs_list_1 = []
    targets_list_1 = []
    inputs_list_2 = []
    targets_list_2 = []

    # Iterate over the first data dictionary
    for key in data_dict_1:
        sample = data_dict_1[key]
        inputs_list_1.append(sample['s_initial_a'])
        targets_list_1.append(sample['s_final'])

    # Iterate over the second data dictionary
    for key in data_dict_2:
        sample = data_dict_2[key]
        inputs_list_2.append(sample['s_initial_a'])
        targets_list_2.append(sample['s_final'])

    # Convert lists to tensors
    X1 = torch.stack(inputs_list_1)
    y1 = torch.stack(targets_list_1)
    X2 = torch.stack(inputs_list_2)
    y2 = torch.stack(targets_list_2)

    # Concatenate tensors from both datasets
    X = torch.cat((X1, X2), dim=0)
    y = torch.cat((y1, y2), dim=0)



    # Create datasets and loaders
    dataset = TensorDataset(X, y)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    model = MLPModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    target_mse= 0.000001
    max_epochs= 1000

    # Train the model
    train_model(model, train_loader, val_loader, device, mse_threshold=target_mse, max_epochs=max_epochs)

    # Load the best model weights
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Prepare a dummy input tensor
    dummy_input = torch.randn(1, 5, device=device)

    # Export the model to ONNX format
    onnx_model_path = 'trained_model.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"Model exported to ONNX format at '{onnx_model_path}'")
