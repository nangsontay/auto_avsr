"""Batched RetinaFace + FAN pipeline.

Replaces the per-frame ibug call loop with two large GPU forward passes per
video. For a 250-frame clip this collapses ~500 kernel launches into ~8.
"""
import cv2
import numpy as np
import torch

from ibug.face_detection.retina_face.prior_box import PriorBox


class BatchedLandmarkPipeline:
    """Drop-in replacement for ibug LandmarksDetector.__call__.

    Accepts a numpy video (N, H, W, 3) uint8 RGB and returns a list of length N
    where each element is either a (68, 2) numpy array of landmarks or None.
    """

    def __init__(self, retina_predictor, fan_predictor,
                 device='cuda:0', det_chunk=64, fan_chunk=128, fp16=True):
        self.face_net    = retina_predictor.net
        self.face_cfg    = retina_predictor.config
        self.face_thresh = retina_predictor.threshold

        self.fan_net = fan_predictor.net
        self.fan_cfg = fan_predictor.config

        self.device     = device
        self.det_chunk  = det_chunk
        self.fan_chunk  = fan_chunk
        self.fp16       = fp16
        self.priors_cache = {}

        self._mean = torch.tensor([104., 117., 123.], device=device).view(1, 3, 1, 1)

        if fp16:
            self.face_net = self.face_net.half()
            # FAN is JIT-traced — need to retrace at fp16, skip for safety
            # FAN forward is fast already in fp32

    def _get_priors(self, h, w):
        key = (h, w)
        if key not in self.priors_cache:
            self.priors_cache[key] = PriorBox(
                self.face_cfg.__dict__, image_size=(h, w)
            ).forward().to(self.device)
        return self.priors_cache[key]

    @torch.no_grad()
    def __call__(self, video_frames_rgb):
        """video_frames_rgb: (N, H, W, 3) uint8 numpy RGB (output of torchcodec)."""
        N, H, W, _ = video_frames_rgb.shape
        if N == 0:
            return []

        # Upload all frames once. (N, H, W, 3) uint8 → ~750 MB for 1000 frames @ 720p
        # For typical 250 frames @ 1080p ~ 1.5 GB peak — fits comfortably.
        frames_bgr = torch.from_numpy(np.ascontiguousarray(video_frames_rgb[..., ::-1])
                                      ).to(self.device, non_blocking=True)

        # ── Stage 1: batched RetinaFace ────────────────────────────────────
        priors    = self._get_priors(H, W)
        var       = self.face_cfg.variance
        scale_box = torch.tensor([W, H, W, H], device=self.device)

        top_boxes_all  = []   # (N, 4)
        top_scores_all = []   # (N,)

        for i in range(0, N, self.det_chunk):
            chunk = frames_bgr[i:i + self.det_chunk]                     # (B, H, W, 3) uint8
            x = chunk.permute(0, 3, 1, 2).float()                        # (B, 3, H, W)
            x = x - self._mean
            if self.fp16:
                x = x.half()
            loc, conf, _ = self.face_net(x)                              # (B, P, 4), (B, P, 2)
            scores = conf[..., 1].float()                                # (B, P)
            top_score, top_idx = scores.max(dim=1)                       # (B,), (B,)

            B = chunk.size(0)
            batch_idx = torch.arange(B, device=self.device)
            top_loc = loc[batch_idx, top_idx].float()                    # (B, 4)
            tp = priors[top_idx]                                         # (B, 4)

            # decode boxes (centre-form → xyxy)
            cx = tp[:, 0] + top_loc[:, 0] * var[0] * tp[:, 2]
            cy = tp[:, 1] + top_loc[:, 1] * var[0] * tp[:, 3]
            w_ = tp[:, 2] * torch.exp(top_loc[:, 2] * var[1])
            h_ = tp[:, 3] * torch.exp(top_loc[:, 3] * var[1])
            boxes = torch.stack([cx - w_/2, cy - h_/2,
                                  cx + w_/2, cy + h_/2], dim=1) * scale_box

            top_boxes_all.append(boxes)
            top_scores_all.append(top_score)

        boxes_gpu  = torch.cat(top_boxes_all, dim=0)                     # (N, 4)
        scores_gpu = torch.cat(top_scores_all, dim=0)                    # (N,)

        boxes_np  = boxes_gpu.cpu().numpy()
        scores_np = scores_gpu.cpu().numpy()

        valid_mask = scores_np >= self.face_thresh
        if not valid_mask.any():
            return [None] * N

        # ── Stage 2: batched FAN ───────────────────────────────────────────
        crop_ratio = self.fan_cfg.crop_ratio                              # 0.55
        input_size = self.fan_cfg.input_size                              # 256

        valid_idx   = np.where(valid_mask)[0]
        face_boxes  = boxes_np[valid_idx]                                 # (V, 4)

        centres        = (face_boxes[:, [0, 1]] + face_boxes[:, [2, 3]]) / 2.0
        face_sizes     = (face_boxes[:, [3, 2]] - face_boxes[:, [1, 0]]).mean(axis=1)
        enl_sizes      = (face_sizes / crop_ratio)[:, None].repeat(2, axis=1)
        enl_boxes      = np.zeros((len(valid_idx), 4), dtype=np.int64)
        enl_boxes[:, :2] = np.round(centres - enl_sizes / 2.0)
        enl_boxes[:, 2:] = np.round(enl_boxes[:, :2] + enl_sizes) + 1

        # Crop + resize each face. cv2.resize is fast (SIMD); doing it on CPU avoids
        # GPU sync overhead for awkward per-face geometry.
        patches = np.empty((len(valid_idx), input_size, input_size, 3), dtype=np.uint8)
        for k, (i, (l, t, r, b)) in enumerate(zip(valid_idx, enl_boxes)):
            img = video_frames_rgb[i]
            pad_l = max(0, -l); pad_t = max(0, -t)
            pad_r = max(0, r - W); pad_b = max(0, b - H)
            if pad_l or pad_t or pad_r or pad_b:
                img = np.pad(img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)))
                lp, tp_, rp, bp = l + pad_l, t + pad_t, r + pad_l, b + pad_t
            else:
                lp, tp_, rp, bp = l, t, r, b
            patches[k] = cv2.resize(img[tp_:bp, lp:rp, :], (input_size, input_size))

        patches_t = torch.from_numpy(patches.transpose(0, 3, 1, 2)
                                     ).to(self.device, non_blocking=True).float() / 255.0

        # Chunked FAN forward
        all_landmarks = []
        for i in range(0, patches_t.size(0), self.fan_chunk):
            sub = patches_t[i:i + self.fan_chunk]
            heatmaps, _, _ = self.fan_net(sub)
            lm = self._decode_fan(heatmaps)                               # (B, 68, 2)
            all_landmarks.append(lm)
        landmarks_dec = torch.cat(all_landmarks, dim=0).cpu().numpy()     # (V, 68, 2)

        hh, hw = input_size // 4, input_size // 4   # FAN heatmap size (typically 64)

        result = [None] * N
        for k, i in enumerate(valid_idx):
            l, t, r, b = enl_boxes[k]
            lm = landmarks_dec[k].copy()
            lm[:, 0] = lm[:, 0] * (r - l) / hw + l
            lm[:, 1] = lm[:, 1] * (b - t) / hh + t
            result[int(i)] = lm

        return result

    def _decode_fan(self, heatmaps):
        """Returns landmark coords (B, K, 2). Mirrors ibug FAN decoder but tensor-only."""
        gamma  = self.fan_cfg.gamma
        radius = self.fan_cfg.radius
        heatmaps = heatmaps.contiguous()

        if (radius ** 2 * heatmaps.shape[2] * heatmaps.shape[3] <
                heatmaps.shape[2] ** 2 + heatmaps.shape[3] ** 2):
            B, K, Hh, Wh = heatmaps.shape
            flat = heatmaps.view(B * K, -1)
            m = flat.argmax(dim=1)
            peak_y = (m // Wh).float()
            peak_x = (m %  Wh).float()
            peaks = torch.stack([peak_y, peak_x], dim=1).view(B, K, 1, 1, 2)

            ys = torch.arange(Hh, device=heatmaps.device).float().view(1, 1, Hh, 1, 1)
            xs = torch.arange(Wh, device=heatmaps.device).float().view(1, 1, 1, Wh, 1)
            grid = torch.cat([ys.expand(1, 1, Hh, Wh, 1),
                              xs.expand(1, 1, Hh, Wh, 1)], dim=4)
            mask = ((grid - peaks).norm(dim=-1) <=
                    radius * (Hh * Wh) ** 0.5).float()
            heatmaps = heatmaps * mask

        x_indices = torch.arange(0.5, heatmaps.shape[3], device=heatmaps.device)
        y_indices = torch.arange(0.5, heatmaps.shape[2], device=heatmaps.device)

        heatmaps = heatmaps.clamp_min(0.0)
        if gamma != 1.0:
            heatmaps = heatmaps.pow(gamma)
        m00 = heatmaps.sum(dim=(2, 3)).clamp_min(torch.finfo(heatmaps.dtype).eps)
        xs = heatmaps.sum(dim=2).mul(x_indices).sum(dim=2).div(m00)
        ys = heatmaps.sum(dim=3).mul(y_indices).sum(dim=2).div(m00)
        return torch.stack([xs, ys], dim=-1)
