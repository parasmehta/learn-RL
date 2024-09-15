import torch

from onnx2torch import convert
from custom_nn import CustomNN, save_model_as_onnx
from custom_env import CustomCartPoleEnv


def load_model_from_onnx(file_name="custom_model.onnx"):
    print("Loading model from ONNX file...")

    model = convert(file_name)
    print("Model: ", model)

    return model


def test_forward_pass(model):
    model = CustomNN()
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 5, dtype=torch.float32)

    # Perform a forward pass
    output = model(dummy_input)

    print("Output shape:", output.shape)
    print("Output:", output)

    save_model_as_onnx(model, "custom_model.onnx")

    # Load the model from the ONNX file
    loaded_model = load_model_from_onnx("custom_model.onnx")

    # Perform a forward pass
    output_loaded_model = loaded_model(dummy_input)

    print("Output_loaded_model shape:", output_loaded_model.shape)
    print("Output_loaded_model:", output_loaded_model)

    assert torch.allclose(output, output_loaded_model, atol=1e-03)


def test_environment():
    # Create the custom environment
    env = CustomCartPoleEnv(render_mode="human")

    # Reset the environment
    state = env.reset()
    print("Initial state:", state)

    terminated = False
    truncated = False

    # Perform a few steps
    while not (terminated or truncated):
        # Take a random action
        action = env.action_space.sample()
        print("Action:", action)

        # Perform the action
        next_state, reward, terminated, truncated, _ = env.step(action)
        print("Next state:", next_state)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)


def save_model():
    # Create the model
    model = CustomNN()
    save_model_as_onnx(model, "custom_model.onnx")


if __name__ == "__main__":

    # Load the model from the ONNX file
    model = load_model_from_onnx("custom_model.onnx")

    # Test the forward pass
    test_forward_pass(model)
    test_environment()
