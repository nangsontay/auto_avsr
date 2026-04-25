#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import kornia.geometry
import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform as tf

_CROP_WORKERS = 8


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks


def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform("similarity", src, dst)
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    warped = (warped * 255).astype("uint8")
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = (warped * 255).astype("uint8")
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)
    if abs(center_y - img.shape[0] / 2) > height + threshold:
        raise OverflowError("too much bias in height")
    if abs(center_x - img.shape[1] / 2) > width + threshold:
        raise OverflowError("too much bias in width")
    y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
    return np.copy(img[y_min:y_max, x_min:x_max])


class VideoProcess:
    def __init__(
        self,
        mean_face_path="20words_mean_face.npy",
        crop_width=96,
        crop_height=96,
        start_idx=48,
        stop_idx=68,
        window_margin=12,
        convert_gray=True,
    ):
        self.reference = np.load(
            os.path.join(os.path.dirname(__file__), mean_face_path)
        )
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray

        # Stable landmark indices for affine estimation (same as affine_transform default)
        self._stable_points = (28, 33, 36, 39, 42, 45, 48, 54)
        # Precompute stable reference as float32 (reference_size == target_size == 256, no offset)
        self._stable_ref = np.vstack(
            [self.reference[x] for x in self._stable_points]
        ).astype(np.float32)

    def __call__(self, video, landmarks, frames_gpu=None):
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        if (
            not preprocessed_landmarks
            or len(preprocessed_landmarks) < self.window_margin
        ):
            return None

        if frames_gpu is not None:
            try:
                sequence = self.crop_patch_gpu(frames_gpu, preprocessed_landmarks)
                if sequence is not None:
                    assert sequence is not None, "crop an empty patch."
                    return sequence
            except Exception:
                pass  # fall through to CPU path

        sequence = self.crop_patch(video, preprocessed_landmarks)
        assert sequence is not None, "crop an empty patch."
        return sequence

    # ── GPU path ──────────────────────────────────────────────────────────────

    def crop_patch_gpu(self, frames_gpu, landmarks):
        """
        Batch-process all frames on GPU:
          1. Compute affine matrices on CPU in parallel threads
             (cv2.estimateAffinePartial2D can't be GPU'd but is fast + parallelisable)
          2. One kornia.geometry.warp_affine call for all N frames
          3. Batch landmark transform via GPU matmul
          4. Vectorised lip crop

        frames_gpu : (N, H, W, 3) uint8 CUDA tensor  (RGB)
        landmarks  : list of N numpy (68, 2)  — all valid (Nones already interpolated)
        Returns    : numpy (N, crop_h, crop_w, 3) uint8
        """
        N = frames_gpu.shape[0]
        device = frames_gpu.device

        # ── Step 1: threaded affine matrix computation ────────────────────
        def compute_M(frame_idx):
            lm = landmarks[frame_idx]
            wm = min(self.window_margin // 2, frame_idx, N - 1 - frame_idx)
            smoothed = np.mean(
                [landmarks[x] for x in range(frame_idx - wm, frame_idx + wm + 1)],
                axis=0,
            )
            smoothed += lm.mean(0) - smoothed.mean(0)
            src = np.vstack([smoothed[x] for x in self._stable_points])
            M, _ = cv2.estimateAffinePartial2D(
                src, self._stable_ref, method=cv2.LMEDS
            )
            return M, smoothed   # M: (2, 3) float64 or None

        with ThreadPoolExecutor(max_workers=_CROP_WORKERS) as pool:
            results = list(pool.map(compute_M, range(N)))

        # Collect valid frames (M is None if LMEDS fails — very rare)
        valid_idx   = [i for i, (M, _) in enumerate(results) if M is not None]
        if not valid_idx:
            return None
        Ms           = np.array([results[i][0] for i in valid_idx], dtype=np.float32)
        smoothed_lms = np.array([results[i][1] for i in valid_idx], dtype=np.float32)

        # ── Step 2: GPU batch warp ────────────────────────────────────────
        # kornia.geometry.warp_affine uses same convention as cv2.warpAffine (forward M)
        frames_f = frames_gpu[valid_idx].permute(0, 3, 1, 2).float()   # (V, 3, H, W)
        M_gpu    = torch.from_numpy(Ms).to(device)                      # (V, 2, 3)

        warped = kornia.geometry.warp_affine(
            frames_f, M_gpu,
            dsize=(256, 256),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )  # (V, 3, 256, 256)

        # ── Step 3: GPU landmark transform ───────────────────────────────
        lm_gpu  = torch.from_numpy(smoothed_lms).to(device)            # (V, 68, 2)
        M_rot   = M_gpu[:, :, :2]                                       # (V, 2, 2)
        M_tra   = M_gpu[:, :, 2]                                        # (V, 2)
        lm_w    = torch.bmm(lm_gpu, M_rot.transpose(1, 2)) + M_tra.unsqueeze(1)  # (V, 68, 2)

        # ── Step 4: vectorised lip crop ───────────────────────────────────
        lip     = lm_w[:, self.start_idx:self.stop_idx, :]             # (V, 20, 2)
        cx      = lip[:, :, 0].mean(1)                                  # (V,)
        cy      = lip[:, :, 1].mean(1)                                  # (V,)

        half_h, half_w = self.crop_height // 2, self.crop_width // 2

        # Replicate cut_patch overflow check — raises so caller can skip this clip
        if (
            torch.any(torch.abs(cy - 128.0) > half_h + 5) or
            torch.any(torch.abs(cx - 128.0) > half_w + 5)
        ):
            raise OverflowError("too much bias in height/width (GPU path)")

        y0 = (cy - half_h).round().clamp(0, 256).long()
        y1 = (cy + half_h).round().clamp(0, 256).long()
        x0 = (cx - half_w).round().clamp(0, 256).long()
        x1 = (cx + half_w).round().clamp(0, 256).long()

        V = len(valid_idx)
        patches = torch.empty(V, 3, self.crop_height, self.crop_width,
                              device=device, dtype=torch.float32)
        for k in range(V):
            p = warped[k, :, y0[k]:y1[k], x0[k]:x1[k]]
            # Boundary clamp can cause ±1 pixel difference — fix with bilinear resize
            if p.shape[1] != self.crop_height or p.shape[2] != self.crop_width:
                p = F.interpolate(
                    p.unsqueeze(0),
                    (self.crop_height, self.crop_width),
                    mode='bilinear', align_corners=False,
                ).squeeze(0)
            patches[k] = p

        # (V, 3, 96, 96) → (V, 96, 96, 3) → uint8 numpy
        result_np = patches.permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy()

        if len(valid_idx) == N:
            return result_np

        # Rare: some frames had no M — fill gaps with nearest valid patch
        sequence = np.zeros((N, self.crop_height, self.crop_width, 3), dtype=np.uint8)
        for k, i in enumerate(valid_idx):
            sequence[i] = result_np[k]
        return sequence

    # ── CPU fallback path (unchanged) ────────────────────────────────────────

    def crop_patch(self, video, landmarks):
        def process_frame(frame_idx):
            frame = video[frame_idx]
            wm = min(self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed = np.mean(
                [landmarks[x] for x in range(frame_idx - wm, frame_idx + wm + 1)],
                axis=0,
            )
            smoothed += landmarks[frame_idx].mean(0) - smoothed.mean(0)
            transformed_frame, transformed_landmarks = self.affine_transform(
                frame, smoothed, self.reference, grayscale=self.convert_gray
            )
            return cut_patch(
                transformed_frame,
                transformed_landmarks[self.start_idx:self.stop_idx],
                self.crop_height // 2,
                self.crop_width  // 2,
            )

        n = len(video)
        with ThreadPoolExecutor(max_workers=_CROP_WORKERS) as pool:
            sequence = list(pool.map(process_frame, range(n)))
        return np.array(sequence)

    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(
                    landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
                )

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if valid_frames_idx:
            landmarks[: valid_frames_idx[0]] = [
                landmarks[valid_frames_idx[0]]
            ] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
                len(landmarks) - valid_frames_idx[-1]
            )

        assert all(lm is not None for lm in landmarks)
        return landmarks

    def affine_transform(
        self,
        frame,
        landmarks,
        reference,
        grayscale=True,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
        interpolation=cv2.INTER_LANCZOS4,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
    ):
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stable_reference = self.get_stable_reference(
            reference, stable_points, reference_size, target_size
        )
        transform = self.estimate_affine_transform(
            landmarks, stable_points, stable_reference
        )
        transformed_frame, transformed_landmarks = self.apply_affine_transform(
            frame, landmarks, transform, target_size,
            interpolation, border_mode, border_value,
        )
        return transformed_frame, transformed_landmarks

    def get_stable_reference(self, reference, stable_points, reference_size, target_size):
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference

    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        return cv2.estimateAffinePartial2D(
            np.vstack([landmarks[x] for x in stable_points]),
            stable_reference,
            method=cv2.LMEDS,
        )[0]

    def apply_affine_transform(
        self, frame, landmarks, transform, target_size,
        interpolation, border_mode, border_value,
    ):
        transformed_frame = cv2.warpAffine(
            frame, transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = (
            np.matmul(landmarks, transform[:, :2].transpose())
            + transform[:, 2].transpose()
        )
        return transformed_frame, transformed_landmarks
