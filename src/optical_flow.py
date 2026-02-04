import os
import cv2 as cv
import numpy as np


def downscale_frame(frame):
    frame = cv.resize(frame, dsize = (320, 320))
    return frame


def upscale_frame(frame):
    frame = cv.resize(frame, dsize=(720, 720))
    return frame


def process_frame_for_optical_flow(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = downscale_frame(frame)
    return frame


def main():
    # read image
    root = os.getcwd()
    video_path = os.path.join(
        root,
        "C:FPVTrickDetector_data/videos_raw/"
        "Flips-_-Rolls-First-Freestyle-flights-Dr_Media_sNyytnm6uio_001_1080p.mp4"
    )
    video_cap_obj = cv.VideoCapture(video_path)

    # get HSV image
    _, frame_first = video_cap_obj.read()
    img_prev = process_frame_for_optical_flow(frame_first)
    img_hsv = np.zeros_like(frame_first)
    img_hsv = downscale_frame(img_hsv)
    img_hsv[:, :, 1] = 255

    while True:
        # read frame
        _, frame_cur = video_cap_obj.read()
        img_cur = process_frame_for_optical_flow(frame_cur)

        # calculate flow
        flow = cv.calcOpticalFlowFarneback(
            prev=img_prev, next=img_cur, flow=None, pyr_scale=.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        mag, ang = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1])

        # draw flow
        img_hsv[:, :, 0] = ang*180 / np.pi / 2
        img_hsv[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        img_bgr = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

        # draw average motion direction
        direction_vectors = []
        for xs, ys in zip(flow[:, :, 0], flow[:, :, 1]):
            for x, y in zip(xs, ys):
                direction_vectors.append([x, y])
        direction_vectors = np.array(direction_vectors)
        avg_direction = -np.sum(direction_vectors, axis=0)
        screen_center = np.array([img_bgr.shape[1] / 2, img_bgr.shape[0] / 2]).astype(int)
        direction_end_point = (screen_center + avg_direction * 0.0001).astype(int)
        img_bgr = cv.line(img_bgr, pt1=screen_center, pt2=direction_end_point, color=[255, 255, 255])

        # display frame
        img_bgr = upscale_frame(img_bgr)
        cv.imshow('Video', img_bgr)
        key = cv.waitKey(16)
        if key == ord('q'):
            break

        img_prev = img_cur


if __name__ == "__main__":
    main()
