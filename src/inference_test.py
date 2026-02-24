from dataclasses import dataclass

import cv2
import mlflow.pytorch
import numpy as np
import torch

from src.dataloaders import LABEL_LOOKUP


VIDEO_NAME = "Back to the hotel - RAW FPV Drone flight [uaLQGt52JJM] 1.mp4"
MODEL = f"models:/FPVTrickDetector/4"


@dataclass
class TrickPopup:
    label: str
    org: list
    lifetime_timer: int
    moving: bool


def draw_text_with_bg(frame, text, org, color=(0, 0, 0), bg_color=(255, 255, 255), font_scale=2, upper=False, thickness=1):
    font_face = cv2.FONT_HERSHEY_PLAIN

    if upper:
        text = text.upper()

    (tw, th), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    pt1 = org.copy()
    pt2 = org.copy()
    pt2[0] += tw
    pt2[1] -= th + baseline

    frame = cv2.rectangle(frame, color=bg_color, thickness=-1, pt1=tuple(pt1), pt2=tuple(pt2))
    frame = cv2.putText(
        frame, text=text, fontFace=font_face,
        fontScale=font_scale, thickness=thickness, color=color, org=tuple(org)
    )

    return frame, tw, th + baseline


def process_trick_popups(trick_popups: list[TrickPopup], frame):
    next_trick_popups = []
    for tp in trick_popups:
        frame, _, _ = draw_text_with_bg(frame, tp.label, tp.org, font_scale=4, upper=True, thickness=2)

        if tp.moving:
            tp.org[1] += 20
            tp.lifetime_timer -= 1
            if tp.lifetime_timer > 0:
                next_trick_popups.append(tp)
        else:
            next_trick_popups.append(tp)

    return next_trick_popups, frame


def main():
    # init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri("file:../mlruns")
    model = mlflow.pytorch.load_model(MODEL)

    # load optical input data
    opt_cap = cv2.VideoCapture("../data/videos_optical_flow/" + VIDEO_NAME)
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
    preds = torch.squeeze(preds)
    frame_labels = torch.argmax(preds, dim=1)

    # load raw video
    raw_cap = cv2.VideoCapture("../data/videos_raw/" + VIDEO_NAME)
    fps = raw_cap.get(cv2.CAP_PROP_FPS)
    frame_step_ms = int(1000. / fps)

    # add preds overlay and display frames
    trick_popups = []
    prev_trick = "none"
    for i, frame_label_tensor in enumerate(frame_labels):
        _, frame = raw_cap.read()

        frame_label = frame_label_tensor.item()

        overlay_org = [100, 100]
        frame_preds = torch.softmax(preds[i], dim=0)
        for label, label_key in LABEL_LOOKUP.items():
            if label_key == frame_label:
                color = (0, 0, 0)
                bg_color = (0, 255, 0)
            else:
                color = (255, 255, 255)
                bg_color = (0, 0, 0)

            frame, tw, th = draw_text_with_bg(frame, label, overlay_org, color, bg_color)

            prob_org = overlay_org.copy()
            prob_org[0] += 100
            label_prob = round(frame_preds[label_key].item(), 2)
            frame, _, _ = draw_text_with_bg(frame, str(label_prob), prob_org, color, bg_color)

            overlay_org[1] += th

        # process trick popups
        curr_trick = [k for k, v in LABEL_LOOKUP.items() if v == frame_label][0]
        if prev_trick != curr_trick:
            if len(trick_popups) > 0:
                trick_popups[-1].moving = True
            if curr_trick != "none":
                trick_popups.append(TrickPopup(curr_trick, [600, 600], 60, False))

        trick_popups, frame = process_trick_popups(trick_popups, frame)
        prev_trick = curr_trick

        cv2.imshow('inference test', frame)

        if cv2.waitKey(frame_step_ms) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
