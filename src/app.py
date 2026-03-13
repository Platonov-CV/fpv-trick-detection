import asyncio
import os
import subprocess
import tempfile
import time
import uuid

import aiofiles
import cv2
import imageio_ffmpeg
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import onnxruntime as ort

from src.dataloaders import LABEL_LOOKUP
from src.inference import draw_text_with_bg, TrickPopup, process_trick_popups
from src.optical_flow import process_frame_for_optical_flow, downscale_frame


MODEL = f"models:/FPVTrickDetector/4"


app = FastAPI()

origins = [
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init model
session = ort.InferenceSession("./model-onnx/model.onnx")

jobs = {}  # job_id → {"progress": [], "done": False, "result_path": None}

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()


@app.post("/process_video/")
async def process_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile | None = None,
    test_video: bool = False,
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"progress": [], "done": False, "result_path": None}
    if not test_video:
        video_bytes = await video_file.read()
    else:
        with open("./data/test/Back to the hotel - RAW FPV Drone flight [uaLQGt52JJM] 1.mp4", 'rb') as file:
            video_bytes = file.read()
    background_tasks.add_task(run_processing, video_bytes, job_id)
    return {"job_id": job_id}


@app.get("/process_video/{job_id}/progress")
async def progress(job_id: str):
    async def stream():
        while not jobs[job_id]["done"]:
            while jobs[job_id]["progress"]:
                msg = jobs[job_id]["progress"].pop(0)
                yield f"data: {msg}\n\n"
            await asyncio.sleep(0.2)
        yield "data: __done__\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/process_video/{job_id}/result")
async def result(job_id: str):
    path = jobs[job_id]["result_path"]
    return StreamingResponse(
        stream_video(path),
        media_type="video/mp4",
        headers={
            "Content-Disposition": "inline; filename=temp_video.mp4",
            "Accept-Ranges": "bytes"  # Important for video seeking
        }
    )


def log_event(job_id, log_str: str):
    print(log_str)
    jobs[job_id]["progress"].append(log_str)


def remove_temp_file(file_path: str):
    for _ in range(10):
        try:
            os.remove(file_path)
            break
        except PermissionError:
            time.sleep(0.1)


async def stream_video(file_path: str):
    """
    An async generator that reads a file in chunks and yields them.
    """
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            # Read and yield file chunks
            while chunk := await f.read(1024):
                yield chunk
    except Exception as e:
        print(f"Error streaming file: {e}")
    finally:
        # Optional: Clean up the temporary file after streaming is complete
        # This can also be done using a background task
        if os.path.exists(file_path):
            remove_temp_file(file_path)


def run_processing(video_bytes: bytes, job_id):
    # save temp raw video
    raw_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    raw_temp.write(video_bytes)
    raw_temp_path = raw_temp.name
    log_event(job_id, "saved temp raw video to " + raw_temp_path)

    # start writing optical flow
    raw_cap = cv2.VideoCapture(raw_temp_path)
    frame_count = int(raw_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, frame_first = raw_cap.read()
    img_prev = process_frame_for_optical_flow(frame_first)
    seq_flow = torch.empty(frame_count, 320, 320, 2)
    img_hsv = np.zeros_like(frame_first)
    img_hsv = downscale_frame(img_hsv)
    img_hsv[:, :, 1] = 255

    # calculate optical flow
    for frame_index in range(frame_count):
        log_event(job_id, "calculating optical flow for frame: " + str(frame_index + 1) + "/" + str(frame_count))

        # read frame
        ret, frame_cur = raw_cap.read()
        if not ret:
            break
        img_cur = process_frame_for_optical_flow(frame_cur)

        # calculate flow
        flow_calc = cv2.calcOpticalFlowFarneback(
            prev=img_prev, next=img_cur, flow=None, pyr_scale=.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        mag, ang = cv2.cartToPolar(flow_calc[:, :, 0], flow_calc[:, :, 1])

        # preprocess
        img_hsv[:, :, 0] = ang * 180 / np.pi / 2
        img_hsv[:, :, 2] = mag
        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        flow_proc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        flow_proc = np.delete(flow_proc, 1, axis=2)
        flow_proc = flow_proc / 255.

        # save to input tensor
        seq_flow[frame_index, :, :, :] = torch.tensor(flow_proc)

        img_prev = img_cur

    seq_flow = torch.moveaxis(seq_flow, 3, 1)
    inputs = seq_flow.unsqueeze(dim=0).numpy()
    log_event(job_id, "calculated optical flow data")

    # predict frame labels
    log_event(job_id, "predicting frame labels...")
    preds = torch.tensor(session.run(["output"], {"input": inputs}))
    preds = torch.squeeze(preds)
    frame_labels = torch.argmax(preds, dim=1)
    log_event(job_id, "predicted frame labels")

    # start writing output video
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    fps = raw_cap.get(cv2.CAP_PROP_FPS)
    opt_temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_temp_path = opt_temp.name
    raw_frame_size = (int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_writer = cv2.VideoWriter(
        out_temp_path,
        fourcc,
        fps,
        raw_frame_size
    )
    raw_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    trick_popups = []
    prev_trick = "none"

    log_event(job_id, "started writing output video...")

    # write output video frame by frame
    for i, frame_label_tensor in enumerate(frame_labels):
        log_event(job_id, "writing output frame: " + str(i + 1) + "/" + str(len(frame_labels)))

        _, frame = raw_cap.read()

        frame_label = frame_label_tensor.item()

        overlay_org = [10, 40]
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
                trick_popups.append(TrickPopup(curr_trick, [10, overlay_org[1] + 40], 60, False))

        trick_popups, frame = process_trick_popups(trick_popups, frame)
        prev_trick = curr_trick

        out_writer.write(frame)

    out_writer.release()
    raw_cap.release()
    remove_temp_file(raw_temp_path)
    log_event(job_id, "finished writing output video")

    # re-encode to H.264
    log_event(job_id, "encoding output video for browser compatibility...")
    out_encoded_temp_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    ffmpeg_result = subprocess.run([
            ffmpeg_path, '-y',
            '-i', out_temp_path,
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',  # required for browser compatibility
            out_encoded_temp_path
        ],
        capture_output=True, text=True
    )
    log_event(job_id, ffmpeg_result.stdout)
    log_event(job_id, ffmpeg_result.stderr)
    remove_temp_file(out_temp_path)
    log_event(job_id, "encoded output video for browser compatibility")

    jobs[job_id]["result_path"] = out_encoded_temp_path
    jobs[job_id]["done"] = True
