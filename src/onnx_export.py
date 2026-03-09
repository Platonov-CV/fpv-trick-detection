import mlflow
import torch.onnx


MODEL = f"models:/FPVTrickDetector/4"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri("file:../mlruns")
model = mlflow.pytorch.load_model(MODEL)

model.eval()
# input shape - (B, T, C, H, W)
# B - batch
# T - frame
# C - channels
# H, W - flow dimensions
dummy_input = torch.randn(1, 1, 2, 320, 320).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "../model-onnx/model.onnx",
    export_params=True,
    input_names=["input"],
    output_names=["output"],
    dynamo=False,
    dynamic_axes={"input": {0: "batch_size", 1: "frame"}, "output": {0: "batch_size", 1: "frame"}}
)
