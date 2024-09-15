import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomNN(nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        # Define a simple feedforward neural network
        # Input layer: 5 inputs (4 state + 1 action)
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)  # Hidden layer: 64 units
        self.fc3 = nn.Linear(64, 4)   # Output layer: 4 outputs (next state)

    def forward(self, state_action):
        # Pass through the network
        x = F.relu(self.fc1(state_action))  # First layer with ReLU activation
        x = F.relu(self.fc2(x))             # Second layer with ReLU activation
        # Output layer, no activation (linear)
        next_state = self.fc3(x)
        return next_state

# Save the model as ONNX


def save_model_as_onnx(model, file_name="dummy_model.onnx"):
    # Create a dummy input for exporting the model
    # Input shape should match the input to the network (5 inputs: 4 state + 1 action)
    dummy_input = torch.randn(1, 5, dtype=torch.float32)

    # Export the model to an ONNX file
    torch.onnx.export(
        model,                       # The model to be exported
        dummy_input,                 # A dummy input for tracing the model structure
        file_name,                   # The output ONNX file name
        export_params=True,          # Store the trained weights inside the model file
        # ONNX opset version (you can change this if needed)
        opset_version=11,
        do_constant_folding=True,    # Whether to fold constants for optimization
        input_names=['state_action'],  # Name of the input tensor
        output_names=['next_state'],  # Name of the output tensor
        dynamic_axes={               # Specify dynamic axes for inputs and outputs
            'state_action': {0: 'batch_size'},
            'next_state': {0: 'batch_size'}
        }
    )
    print(f"Model has been successfully saved as {file_name}")
