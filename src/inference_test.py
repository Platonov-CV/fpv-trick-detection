import cv2
import mlflow.pytorch
import numpy as np
import torch

from src.dataloaders import LABEL_LOOKUP
from src.model import FPVTrickDetector


# init model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri("file:../mlruns")
model = mlflow.pytorch.load_model(f"models:/FPVTrickDetector/2")

# load optical input data
opt_cap = cv2.VideoCapture("../data/inference_test/Back to the hotel - RAW FPV Drone flight [uaLQGt52JJM] 1 _opt.mp4")
frame_count = int(opt_cap.get(cv2.CAP_PROP_FRAME_COUNT))

seq_flow = torch.empty(frame_count, 320, 320, 2).half()

for frame_index in range(frame_count):
    _, frame_cur = opt_cap.read()

    flow = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2HSV)
    flow = np.delete(flow, 1, axis=2)
    flow = flow / 255.

    seq_flow[frame_index, :, :, :] = torch.tensor(flow)

seq_flow = torch.moveaxis(seq_flow, 3, 1)
inputs = seq_flow.unsqueeze(dim=0).to(device)

opt_cap.release()

# predict frame labels
with torch.inference_mode():
    with torch.amp.autocast("cuda"):
        preds = model(inputs)
frame_labels = torch.argmax(torch.squeeze(preds), dim=1)

# load raw video
raw_cap = cv2.VideoCapture("../data/inference_test/Back to the hotel - RAW FPV Drone flight [uaLQGt52JJM] 1 _raw.mp4")
fps = raw_cap.get(cv2.CAP_PROP_FPS)
frame_step_ms = int(1000. / fps)

# display video with frame labels overlay
for frame_label_tensor in frame_labels:
    _, frame = raw_cap.read()

    frame_label = frame_label_tensor.item()
    label_str = [k for k, v in LABEL_LOOKUP.items() if v == frame_label][0]

    frame = cv2.rectangle(frame, color=(0, 0, 0), thickness=-1, pt1=(100, 100), pt2=(150, 50))
    frame = cv2.putText(
        frame, text=label_str, fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1, thickness=1, color=(255, 255, 255), org=(100, 100)
    )

    cv2.imshow('inference test', frame)

    if cv2.waitKey(frame_step_ms) & 0xFF == ord('q'):
        break