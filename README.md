# FPV trick detection

The goal of this project was to develop a custom model for classifying frames of FPV freestyle recording videos by the trick being performed.

## Goal

Model should be able to correctly label frames from video containing following tricks:

<table>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/cbb8c509-5cfa-4b48-ad23-cad7c36a6ca3" width="200"><br>roll</td>
    <td align="center"><img src="https://github.com/user-attachments/assets/1b19f219-41a8-4b56-9660-0dd06572fb4a" width="200"><br>flip</td>
    <td align="center"><img src="https://github.com/user-attachments/assets/2cf3ec56-4344-4fcc-a709-8b83eff3da7c" width="200"><br>spin</td>
  </tr>
</table>

## Overview

Core pipeline goes as follows:
1. Calculate optical flow for each frame using Farneback optical flow algorithm from OpenCV
2. Model inference on the whole frame sequence

Model architecture consists of two blocks:
- CNN - for extracting motion features from each optical flow frame
- bidirectional GRU - for modeling temporal relations between optical flow frame sequences and ground truth trick labels

<img width="500" alt="fpv-trick-detection_pipeline-diagram drawio" src="https://github.com/user-attachments/assets/5750156f-20df-4ec9-b870-c378cca6806a" />

## Training data

30 recordings of FPV freestyle were collected from Youtube and manually annotated using VIA Video Annotator.

The videos were specifically filtered to be uploaded before 2018, as that period of FPV freestyle featured very distinct few tricks that are easily classifiable.

Before the training process optical flow from collected videos was preprocessed and saved to speed up the training process and remove the unnecessary recalculation of the optical flow that doesn't feature any learnable parameters.

> RAW VIDEO FRAGMENT

> OPTICAL FLOW FRAGMENT

## Optimization

The training data featured heavy class imbalance, which necessitated using weighted cross entropy loss for model optimization.

> CLASS BALANCE HISTOGRAM

Training process was logged using MLFlow, which helped to optimize various architecture features in a reliable and transparent way.

## Results

> metrics
> 
The model learned to classify tricks in a very stable and predictable manner, while making mistakes only in ambiguous cases that are hard to classify even for a human.

## Deployment

Model was converted to ONNX format for more lightweight and compatible inference. The whole pipeline including inference and other supporting processing was containerized with Docker for use in compose as a microservice.

Running container functions as a FastAPI server, that receives raw video and returns back rerendered video with overlayed trick probabilities for each frame.

## Utilized technologies

- PyTorch
- MLFlow
- Docker
- ONNX
- OpenCV
- VIA Video Annotator
- FastAPI
