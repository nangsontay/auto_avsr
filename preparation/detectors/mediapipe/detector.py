#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_MODEL_PATH = "/tmp/blaze_face_short_range.tflite"

# Keypoint indices in mediapipe FaceDetector result
# 0: right eye, 1: left eye, 2: nose tip, 3: mouth center, 4: right ear, 5: left ear
_RIGHT_EYE = 0
_LEFT_EYE = 1
_NOSE = 2
_MOUTH = 3


class LandmarksDetector:
    def __init__(self):
        options_short = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            min_detection_confidence=0.5,
        )
        options_full = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            min_detection_confidence=0.5,
        )
        self.short_range_detector = vision.FaceDetector.create_from_options(options_short)
        self.full_range_detector = vision.FaceDetector.create_from_options(options_full)

    def __call__(self, video_frames):
        landmarks = self.detect(video_frames, self.full_range_detector)
        if all(element is None for element in landmarks):
            landmarks = self.detect(video_frames, self.short_range_detector)
            assert any(l is not None for l in landmarks), "Cannot detect any frames in the video"
        return landmarks

    def detect(self, video_frames, detector):
        import mediapipe as mp
        landmarks = []
        for frame in video_frames:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = detector.detect(mp_image)
            if not result.detections:
                landmarks.append(None)
                continue
            # Pick largest face
            best = max(result.detections, key=lambda d: (
                d.bounding_box.width * d.bounding_box.height
            ))
            ih, iw = frame.shape[:2]
            kps = best.keypoints
            lmx = np.array([
                [kps[_RIGHT_EYE].x * iw, kps[_RIGHT_EYE].y * ih],
                [kps[_LEFT_EYE].x * iw,  kps[_LEFT_EYE].y * ih],
                [kps[_NOSE].x * iw,       kps[_NOSE].y * ih],
                [kps[_MOUTH].x * iw,      kps[_MOUTH].y * ih],
            ])
            landmarks.append(lmx)
        return landmarks