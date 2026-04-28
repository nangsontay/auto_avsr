"""Batched RetinaFace + FAN pipeline.

Replaces the per-frame ibug call loop with two large GPU forward passes per
video. For a 250-frame clip this collapses ~500 kernel launches into ~8.
"""
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

from ibug.face_detection.retina_face.prior_box import PriorBox

# Module-level thread pool for face-crop resizes (cv2 releases GIL).
# Sized adaptively: scale with CPU count but cap to avoid oversubscription on
# small machines. On 8-core: 4 thread; on 32-core: 8 thread (cap).
import os as _os
_CROP_POOL_SIZE = max(2, min(8, (_os.cpu_count() or 4) // 2))
_FACE_CROP_POOL = ThreadPoolExecutor(max_workers=_CROP_POOL_SIZE)


def _build_buckets(max_chunk: int, levels: int = 4, min_size: int = 32):
    """Build a small set of fixed batch-size buckets from `max_chunk` down to
    `min_size` by halving. Returns ascending list, e.g. for max_chunk=512:
    [64, 128, 256, 512]. The bucket scheme caps unique CUDAGraph shapes per
    compiled net to len(buckets), keeping CUDAGraph replay benefits while
    bounding padding waste to <2x worst-case."""
    buckets, b = [], int(max_chunk)
    for _ in range(levels):
        if b < min_size:
            break
        buckets.append(b)
        b //= 2
    buckets = sorted(set(buckets))
    if not buckets or buckets[0] > min_size:
        buckets.insert(0, min_size)
    return buckets


def _next_bucket(n: int, buckets) -> int:
    """Smallest bucket >= n. If n exceeds the largest bucket, return n itself
    (caller must already have ensured n <= max_chunk via the chunked loop)."""
    for b in buckets:
        if b >= n:
            return b
    return n


class BatchedLandmarkPipeline:
    def __init__(self, retina_predictor, fan_predictor,
                 device='cuda:0', det_chunk=96, fan_chunk=192,
                 fp16=True, use_compile=True):
        self.face_net    = retina_predictor.net
        self.face_cfg    = retina_predictor.config
        self.face_thresh = retina_predictor.threshold
        self.fan_net     = fan_predictor.net
        self.fan_cfg     = fan_predictor.config
        self.device      = device
        self.det_chunk   = det_chunk
        self.fan_chunk   = fan_chunk
        self.fp16        = fp16
        self.priors_cache = {}

        # Bucket sizes for ragged tail batches. Padding to one of a few fixed
        # shapes (instead of pad-to-max) caps CUDAGraph re-records to len(buckets)
        # per compiled net, while keeping padding waste bounded (<2x worst case,
        # ~25% average) — see PyTorch CUDAGraph dynamic-shape recommendation.
        self.det_buckets = _build_buckets(det_chunk, levels=4)
        self.fan_buckets = _build_buckets(fan_chunk, levels=4)

        self._mean = torch.tensor([104., 117., 123.], device=device).view(1, 3, 1, 1)

        if fp16:
            self.face_net = self.face_net.half()

        if use_compile:
            try:
                self.face_net = torch.compile(self.face_net, mode='reduce-overhead')
                # FAN is JIT-traced; compile the underlying net only if not already traced
                if not isinstance(fan_predictor.net, torch.jit.ScriptModule):
                    self.fan_net = torch.compile(self.fan_net, mode='reduce-overhead')
            except Exception:
                pass  # torch.compile not available or failed — run eager

    def _get_priors(self, h, w):
        key = (h, w)
        if key not in self.priors_cache:
            self.priors_cache[key] = PriorBox(
                self.face_cfg.__dict__, image_size=(h, w)
            ).forward().to(self.device)
        return self.priors_cache[key]

    @torch.no_grad()
    def __call__(self, video_frames_rgb):
        """video_frames_rgb: (N, H, W, 3) uint8 numpy RGB."""
        N, H, W, _ = video_frames_rgb.shape
        if N == 0:
            return [], None

        # Upload RGB directly — flip channels on GPU to avoid a full CPU copy
        frames_gpu = torch.from_numpy(video_frames_rgb).to(
            self.device, non_blocking=True
        )                                                                   # (N, H, W, 3) uint8
        frames_bgr = frames_gpu.flip(-1)                                    # RGB→BGR on GPU, zero copy

        # ── Stage 1: batched RetinaFace ────────────────────────────────────
        priors    = self._get_priors(H, W)
        var       = self.face_cfg.variance
        scale_box = torch.tensor([W, H, W, H], device=self.device)

        top_boxes_all  = []
        top_scores_all = []

        # Adaptive chunk: nếu RetinaFace forward OOM, chia đôi det_chunk cho phần còn lại.
        i = 0
        det_chunk = self.det_chunk
        while i < N:
            chunk = frames_bgr[i:i + det_chunk]
            B = chunk.size(0)
            # Round up to the nearest bucket so torch.compile/CUDAGraph re-uses
            # one of len(self.det_buckets) recorded graphs (instead of one per
            # unique tail size). Skip padding when B already matches a bucket.
            target = _next_bucket(B, self.det_buckets)
            try:
                if target > B:
                    pad_n = target - B
                    chunk_padded = torch.cat(
                        [chunk, chunk[-1:].expand(pad_n, -1, -1, -1)], dim=0
                    )
                else:
                    chunk_padded = chunk
                x = chunk_padded.permute(0, 3, 1, 2).float() - self._mean
                if self.fp16:
                    x = x.half()
                loc, conf, _ = self.face_net(x)
                # Drop padding rows before downstream math
                loc = loc[:B]
                conf = conf[:B]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if det_chunk == 1:
                    raise
                det_chunk = max(1, det_chunk // 2)
                continue
            scores = conf[..., 1].float()
            top_score, top_idx = scores.max(dim=1)

            bi = torch.arange(B, device=self.device)
            top_loc = loc[bi, top_idx].float()
            tp = priors[top_idx]

            cx = tp[:, 0] + top_loc[:, 0] * var[0] * tp[:, 2]
            cy = tp[:, 1] + top_loc[:, 1] * var[0] * tp[:, 3]
            w_ = tp[:, 2] * torch.exp(top_loc[:, 2] * var[1])
            h_ = tp[:, 3] * torch.exp(top_loc[:, 3] * var[1])
            boxes = torch.stack([cx - w_/2, cy - h_/2,
                                  cx + w_/2, cy + h_/2], dim=1) * scale_box
            top_boxes_all.append(boxes)
            top_scores_all.append(top_score)
            i += det_chunk

        boxes_np  = torch.cat(top_boxes_all).cpu().numpy()
        scores_np = torch.cat(top_scores_all).cpu().numpy()

        valid_mask = scores_np >= self.face_thresh
        if not valid_mask.any():
            return [None] * N, frames_gpu

        # ── Stage 2: batched FAN ───────────────────────────────────────────
        crop_ratio = self.fan_cfg.crop_ratio
        input_size = self.fan_cfg.input_size

        valid_idx  = np.where(valid_mask)[0]
        face_boxes = boxes_np[valid_idx]

        centres   = (face_boxes[:, [0, 1]] + face_boxes[:, [2, 3]]) / 2.0
        face_sizes = (face_boxes[:, [3, 2]] - face_boxes[:, [1, 0]]).mean(axis=1)
        enl_sizes  = (face_sizes / crop_ratio)[:, None].repeat(2, axis=1)
        enl_boxes  = np.zeros((len(valid_idx), 4), dtype=np.int64)
        enl_boxes[:, :2] = np.round(centres - enl_sizes / 2.0)
        enl_boxes[:, 2:] = np.round(enl_boxes[:, :2] + enl_sizes) + 1

        # Pull frames back to CPU once for crop (cheaper than keeping them on GPU
        # given the non-uniform per-face padding/resize geometry)
        frames_cpu = video_frames_rgb  # already numpy, no copy

        patches = np.empty((len(valid_idx), input_size, input_size, 3), dtype=np.uint8)

        def crop_one(args):
            k, i, (l, t, r, b) = args
            img = frames_cpu[i]
            pad_l = max(0, -l); pad_t = max(0, -t)
            pad_r = max(0, r - W); pad_b = max(0, b - H)
            if pad_l or pad_t or pad_r or pad_b:
                img = np.pad(img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)))
                lp, tp_, rp, bp = l+pad_l, t+pad_t, r+pad_l, b+pad_t
            else:
                lp, tp_, rp, bp = l, t, r, b
            patches[k] = cv2.resize(img[tp_:bp, lp:rp], (input_size, input_size))

        # Parallel crop — cv2 releases GIL
        list(_FACE_CROP_POOL.map(
            crop_one,
            [(k, int(i), enl_boxes[k]) for k, i in enumerate(valid_idx)]
        ))

        patches_t = torch.from_numpy(
            patches.transpose(0, 3, 1, 2).copy()
        ).to(self.device, non_blocking=True).float() / 255.0

        # Chunked FAN forward — với adaptive OOM retry.
        # Nếu một chunk gây CUDA OOM, ta empty_cache() và chia đôi chunk size cho
        # phần còn lại của video. Chỉ raise khi đã chunk = 1 mà vẫn OOM (lúc đó
        # GPU thực sự không đủ VRAM cho cả 1 patch — không cứu được).
        all_lm = []
        i = 0
        chunk = self.fan_chunk
        N_patch = patches_t.size(0)
        while i < N_patch:
            sub = patches_t[i:i + chunk]
            real_b = sub.size(0)
            # Round up to the nearest FAN bucket so CUDAGraph re-uses one of
            # len(self.fan_buckets) recorded graphs.
            target = _next_bucket(real_b, self.fan_buckets)
            try:
                if target > real_b:
                    pad_n = target - real_b
                    sub_padded = torch.cat(
                        [sub, sub[-1:].expand(pad_n, -1, -1, -1)], dim=0
                    )
                else:
                    sub_padded = sub
                heatmaps, _, _ = self.fan_net(sub_padded)
                # Drop padding rows before decoding
                all_lm.append(self._decode_fan(heatmaps[:real_b]))
                i += chunk
            except torch.cuda.OutOfMemoryError:
                # Giải phóng buffer và thử với chunk nhỏ hơn
                torch.cuda.empty_cache()
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
        landmarks_dec = torch.cat(all_lm).cpu().numpy()

        hh = hw = input_size // 4  # FAN heatmap stride = 4

        result = [None] * N
        for k, i in enumerate(valid_idx):
            l, t, r, b = enl_boxes[k]
            lm = landmarks_dec[k].copy()
            lm[:, 0] = lm[:, 0] * (r - l) / hw + l
            lm[:, 1] = lm[:, 1] * (b - t) / hh + t
            result[int(i)] = lm

        # Return landmarks AND the RGB GPU tensor so the caller can
        # reuse it for GPU crop_patch without a second host→device copy.
        return result, frames_gpu

    def _decode_fan(self, heatmaps):
        gamma  = self.fan_cfg.gamma
        radius = self.fan_cfg.radius
        heatmaps = heatmaps.contiguous()

        if (radius ** 2 * heatmaps.shape[2] * heatmaps.shape[3] <
                heatmaps.shape[2] ** 2 + heatmaps.shape[3] ** 2):
            B, K, Hh, Wh = heatmaps.shape
            m = heatmaps.view(B * K, -1).argmax(1)
            peak_y = (m // Wh).float()
            peak_x = (m %  Wh).float()
            peaks = torch.stack([peak_y, peak_x], dim=1).view(B, K, 1, 1, 2)
            ys = torch.arange(Hh, device=heatmaps.device).float().view(1,1,Hh,1,1)
            xs = torch.arange(Wh, device=heatmaps.device).float().view(1,1,1,Wh,1)
            grid = torch.cat([ys.expand(1,1,Hh,Wh,1), xs.expand(1,1,Hh,Wh,1)], dim=4)
            mask = ((grid - peaks).norm(dim=-1) <=
                    radius * (Hh * Wh) ** 0.5).float()
            heatmaps = heatmaps * mask

        x_idx = torch.arange(0.5, heatmaps.shape[3], device=heatmaps.device)
        y_idx = torch.arange(0.5, heatmaps.shape[2], device=heatmaps.device)
        heatmaps = heatmaps.clamp_min(0.0)
        if gamma != 1.0:
            heatmaps = heatmaps.pow(gamma)
        m00 = heatmaps.sum(dim=(2, 3)).clamp_min(torch.finfo(heatmaps.dtype).eps)
        xs = heatmaps.sum(dim=2).mul(x_idx).sum(dim=2).div(m00)
        ys = heatmaps.sum(dim=3).mul(y_idx).sum(dim=2).div(m00)
        return torch.stack([xs, ys], dim=-1)
