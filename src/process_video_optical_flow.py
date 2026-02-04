import os.path
from pathlib import Path

import cv2
import numpy as np

from src.optical_flow import process_frame_for_optical_flow, downscale_frame


raw_video_paths = [f for f in Path("../data/videos_raw").iterdir()]
processed_video_paths = [f for f in Path("../data/videos_optical_flow").iterdir()]

# leave only the unprocessed videos
videos_to_process = [
    raw_path for raw_path in raw_video_paths if raw_path.name not in [
        processed_path.name for processed_path in processed_video_paths
    ]
]

for path in videos_to_process:
    cap = cv2.VideoCapture(path)

    fourcc = cv2.VideoWriter.fourcc('m','p','4','v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(
        os.path.join(Path("../data/videos_optical_flow/"), path.name),
        fourcc,
        fps,
        (320, 320)
    )

    # get HSV image
    _, frame_first = cap.read()
    img_prev = process_frame_for_optical_flow(frame_first)
    img_hsv = np.zeros_like(frame_first)
    img_hsv = downscale_frame(img_hsv)
    img_hsv[:, :, 1] = 255

    frame_index = 0

    while True:
        print("Calculating optical flow for frame: ", frame_index, ", ", path)

        # read frame
        ret, frame_cur = cap.read()
        if not ret:
            break
        img_cur = process_frame_for_optical_flow(frame_cur)

        # calculate flow
        flow = cv2.calcOpticalFlowFarneback(
            prev=img_prev, next=img_cur, flow=None, pyr_scale=.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])

        # draw flow
        img_hsv[:, :, 0] = ang * 180 / np.pi / 2
        img_hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        # write optical flow frame to file
        out.write(img_bgr)

        img_prev = img_cur
        frame_index += 1

    cap.release()
    out.release()