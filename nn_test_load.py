import numpy as np
import onnxruntime as ort

# Load the ONNX model
onnx_model_path = 'trained_model.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# Prepare input data
X_test = np.random.rand(10, 5).astype(np.float32)

# Run inference
outputs = ort_session.run(
    None,                  # Output names (None means all outputs)
    {'input': X_test}      # Input name and data
)

# Get the output
predictions = outputs[0]
print(predictions)
