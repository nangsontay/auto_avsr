#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings

import torch

from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor

warnings.filterwarnings("ignore")


def _is_volta_gpu(device: str) -> bool:
    """Return True if the given CUDA device is a Volta-class GPU (sm_70).

    cuDNN 9.5+ removed several precompiled convolution engines for sm_70,
    causing `torch.jit.trace` inside FANPredictor to raise
    "RuntimeError: FIND was unable to find an engine to execute this computation".
    We detect Volta and turn cuDNN off only for the trace step.
    """
    if not torch.cuda.is_available():
        return False
    if not isinstance(device, str) or not device.startswith("cuda"):
        return False
    try:
        idx = int(device.split(":", 1)[1]) if ":" in device else 0
    except (ValueError, IndexError):
        idx = 0
    try:
        major, minor = torch.cuda.get_device_capability(idx)
    except Exception:
        return False
    return (major, minor) == (7, 0)


class LandmarksDetector:
    def __init__(self, device="cuda:0", model_name="mobilenet0.25"):
        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
        )

        # Defensive guard: on Volta (V100 / Titan V / Quadro GV100) the cuDNN 9
        # engine finder can fail during torch.jit.trace inside FANPredictor.
        # Disable cuDNN only for the trace, then restore it so other models
        # (e.g. RetinaFace already constructed above) keep full performance.
        prev_cudnn_enabled = torch.backends.cudnn.enabled
        if _is_volta_gpu(device):
            torch.backends.cudnn.enabled = False
        try:
            self.landmark_detector = FANPredictor(device=device, model=None)
        finally:
            torch.backends.cudnn.enabled = prev_cudnn_enabled

    def __call__(self, video_frames):
        landmarks = []
        for frame in video_frames:
            detected_faces = self.face_detector(frame, rgb=False)
            face_points, _ = self.landmark_detector(frame, detected_faces, rgb=True)
            if len(detected_faces) == 0:
                landmarks.append(None)
            else:
                max_id, max_size = 0, 0
                for idx, bbox in enumerate(detected_faces):
                    bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                    if bbox_size > max_size:
                        max_id, max_size = idx, bbox_size
                landmarks.append(face_points[max_id])
        return landmarks
