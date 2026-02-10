from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from ast import literal_eval


label_lookup = {
    'in_control': 0,
    'crash': 1,
}


class FPVCrashDataset(Dataset):
    def __init__(self):
        self.video_paths = [f for f in Path("../data/videos_optical_flow").iterdir()]

        self.label_segments = pd.read_csv(
            "../data/labels.csv", header=1
        )
        self.label_segments['file_list'] = self.label_segments['file_list'].apply(literal_eval)
        self.label_segments['metadata'] = self.label_segments['metadata'].apply(literal_eval)
        self.label_segments['filename'] = self.label_segments['file_list'].apply(lambda l: l[0])
        self.label_segments = self.label_segments.dropna()


    def __len__(self):
        return len(self.video_paths)


    def __getitem__(self, item):
        # load optical flow data from video file
        path = self.video_paths[item]

        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        seq_flow = torch.empty(frame_count, 320, 320, 2).half()

        for frame_index in range(frame_count):
            _, frame_cur = cap.read()

            flow = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2HSV)
            flow = np.delete(flow, 1, axis=2)
            flow = flow / 255.

            seq_flow[frame_index, :, :, :] = torch.tensor(flow)

        seq_flow = torch.moveaxis(seq_flow, 3, 1)

        # load frame labels from df
        label_segments = self.label_segments[path.name == self.label_segments["filename"]]
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_step = 1. / fps
        segment_index = 0
        cur_segment = label_segments.iloc[segment_index]
        labels = []
        for frame_index in range(0, frame_count):
            if frame_index * frame_step > cur_segment.temporal_segment_end:
                segment_index += 1
                cur_segment = label_segments.iloc[segment_index]
            label = cur_segment.metadata["TEMPORAL-SEGMENTS"]
            label_int = label_lookup[label]
            labels.append(label_int)
        labels = torch.tensor(labels)

        cap.release()
        return seq_flow, labels


def get_dataloaders():
    ds = FPVCrashDataset()

    train_ds, val_ds = random_split(ds, [.8, .2], generator=torch.Generator().manual_seed(777))

    train_dl = DataLoader(train_ds, batch_size=1)
    val_dl = DataLoader(val_ds, batch_size=1)

    return train_dl, val_dl


def main():
    train_dl, val_dl = get_dataloaders()
    item = next(iter(train_dl))
    item = next(iter(val_dl))


if __name__ == "__main__":
    main()
