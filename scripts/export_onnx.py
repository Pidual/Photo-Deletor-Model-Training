from pathlib import Path

import torch
from torchvision.models import resnet50, ResNet50_Weights

# 1. Define the parameters you used during training
num_classes = 2
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PATH = MODELS_DIR / "resnet50_finetunedV2.pth"
ONNX_PATH = MODELS_DIR / "resnet50_classifier.onnx"

# 2. Re-instantiate the base model with the exact same weights argument
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# 3. Re-apply the exact same final layer modification
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 4. Load the trained weights from the file
model.load_state_dict(torch.load(PATH, map_location="cpu"))

# 5. Set to evaluation mode (keep on CPU for ONNX export)
model.eval()

# Create dummy input matching your input size
dummy_input = torch.randn(1, 3, 512, 512)

# Export to ONNX using legacy exporter (more compatible)
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["image"],
    output_names=["prediction"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "prediction": {0: "batch_size"}
    },
    opset_version=14,
    do_constant_folding=True,
    export_params=True,
    dynamo=False,  # Use legacy exporter to avoid Python 3.14 compatibility issues
)

print("✅ Model exported to ONNX format!")

# Optional: Verify the ONNX model
try:
    import onnx
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid!")
except ImportError:
    print("⚠️ Install onnx package to verify: pip install onnx")
except Exception as e:
    print(f"❌ ONNX validation error: {e}")