import csv
import os
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from glob import glob
import multiprocessing
from multiprocessing import Manager, Pool
from pathlib import Path

import av
import cv2
import numpy as np
import psutil
import torch
from sklearn.model_selection import train_test_split
from torchcodec.decoders import VideoDecoder

from transforms import TextTransform

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = '/app/dataset'
OUTPUT_DIR   = '/app/vnlr'
LABELS_DIR   = '/app/labels'
DONE_LOG     = '/app/resume_state/preprocess_done.txt'
OOM_LOG      = '/app/resume_state/preprocess_oom_retry.txt'  # files OOM-deferred; run 1-worker retry pass

NUM_GPUS          = 1
WORKERS_PER_GPU   = 1     # 2 workers per GPU. Trên GPU 16 GB cần chunk nhỏ — xem dưới.
DECODE_THREADS    = 4     # per worker — mostly I/O wait, ok to oversubscribe
CROP_WORKERS      = 6     # per worker — pipeline stage: GPU runs while these crop previous video
SAVE_THREADS      = 3     # per worker
PREFETCH_Q        = 16    # decoded frames waiting for GPU
CROP_Q_DEPTH      = 8     # (frames, landmarks) pairs waiting for crop workers
SAVE_Q_DEPTH      = 16    # cropped videos waiting to save
# Chunk size cho normal pass (2 worker chia VRAM). Nếu GPU 24 GB có thể nâng lên 128 / 192.
DET_CHUNK         = 256    # RetinaFace batch — nhỏ vừa cho 2 worker share GPU 16 GB
FAN_CHUNK         = 512    # FAN batch — chunk peak VRAM cao nhất, cần giữ thấp khi share GPU
# Chunk size cho retry pass (1 worker, full VRAM, video bị OOM trước đó)
RETRY_DET_CHUNK   = 32    # rất bảo thủ — clip bị OOM thường rất dài/HD
RETRY_FAN_CHUNK   = 48
USE_FP16          = True
USE_COMPILE       = True # CUDA Graphs giữ private memory pool gây fragmentation khi share GPU
USE_NVENC         = True # h264_nvenc tốn 200-500 MiB VRAM trên cùng GPU đang inference
# OOM-retry behaviour ─────────────────────────────────────────────
OOM_MAX_ATTEMPTS  = 3     # số lần thử lại 1 video khi gặp CUDA OOM trước khi defer sang retry pass
OOM_BACKOFF_SEC   = 1.5   # giây chờ giữa các lần retry (cho sibling worker thoát đỉnh peak)
# ──────────────────────────────────────────────────────────────────────────────


def load_video(path):
    return VideoDecoder(path).get_all_frames().data.permute(0, 2, 3, 1).numpy()


# Tested once at startup; workers read this flag (set in __main__)
_NVENC_OK: bool = True


def _write_video_pyav(filename, vid, fps, codec, options):
    """Write via PyAV (no subprocess spawn overhead, ~1.3x faster than ffmpeg pipe)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    h, w = vid.shape[1], vid.shape[2]
    container = av.open(filename, 'w')
    stream = container.add_stream(codec, rate=fps)
    stream.width, stream.height = w, h
    stream.pix_fmt = 'yuv420p'
    stream.options = options
    for frame_np in vid:
        frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def save2vid(filename, vid, fps):
    """Try NVENC first (GPU 0 — CUDA_VISIBLE_DEVICES already virtualises the right GPU),
    fall back to libx264 ultrafast on failure. Both paths use PyAV — no subprocess spawn."""
    if _NVENC_OK:
        try:
            # '-gpu 0' always: each worker has CUDA_VISIBLE_DEVICES set so its GPU IS index 0
            _write_video_pyav(filename, vid, fps,
                              'h264_nvenc', {'preset': 'p1', 'gpu': '0'})
            return
        except Exception:
            pass
    _write_video_pyav(filename, vid, fps,
                      'libx264', {'preset': 'ultrafast', 'crf': '23'})


_SENTINEL = object()


def _is_cuda_oom(exc):
    """Return True if `exc` represents a CUDA out-of-memory condition.

    Catches both:
      - `torch.cuda.OutOfMemoryError` (raw eager-mode OOM)
      - `RuntimeError` re-wrapped by the TorchScript interpreter when OOM
        happens inside a scripted/compiled module (e.g. fan_net). The wrapped
        message still contains 'CUDA out of memory' / 'out of memory'.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return ('out of memory' in msg) or ('cuda error: out of memory' in msg)
    return False


def _infer_with_oom_retry(fast_pipeline, frames,
                          max_attempts=None, backoff_sec=None):
    """Run fast_pipeline(frames) with adaptive OOM retry.

    Strategy on torch.cuda.OutOfMemoryError (or RuntimeError-wrapped OOM from
    TorchScript):
      1. empty_cache + synchronize to release any partial allocations.
      2. Sleep `backoff_sec * (attempt+1)` so the sibling worker on the same
         GPU has time to drain its own peak (when 2 workers/GPU contend, the
         OOM is usually transient — peaks rarely overlap for long).
      3. Halve det_chunk and fan_chunk for the next attempt (mutating the
         pipeline in place; restored in finally).

    Raises the original OOM exception if all attempts fail — caller should
    defer the file to the OOM retry log for a single-worker pass at the end.
    Non-OOM RuntimeErrors are propagated immediately without retry.
    """
    if max_attempts is None:
        max_attempts = OOM_MAX_ATTEMPTS
    if backoff_sec is None:
        backoff_sec = OOM_BACKOFF_SEC

    original_det = fast_pipeline.det_chunk
    original_fan = fast_pipeline.fan_chunk
    last_exc = None
    try:
        for attempt in range(max_attempts):
            try:
                return fast_pipeline(frames)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # Chỉ retry nếu thực sự là OOM (kể cả khi bị TorchScript bọc
                # thành RuntimeError). RuntimeError không phải OOM phải ném
                # ra ngoài luôn để caller xử lý đúng (INFER-ERR).
                if not _is_cuda_oom(e):
                    raise
                last_exc = e
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass
                if attempt < max_attempts - 1:
                    time.sleep(backoff_sec * (attempt + 1))
                    fast_pipeline.det_chunk = max(8, fast_pipeline.det_chunk // 2)
                    fast_pipeline.fan_chunk = max(8, fast_pipeline.fan_chunk // 2)
        raise last_exc
    finally:
        fast_pipeline.det_chunk = original_det
        fast_pipeline.fan_chunk = original_fan


def worker(worker_id, gpu_id, file_chunk, train_set, val_set, test_set,
           done_set, lock, counters, start_time, retry_mode=False,
           total_workers=None, nvenc_ok=True):
    # ── NVENC availability (BUG 2 fix) ─────────────────────────────────────
    # 'spawn' workers re-import the module so module-level _NVENC_OK is reset
    # to the default True. We propagate the parent's probe result here.
    global _NVENC_OK
    _NVENC_OK = bool(nvenc_ok)

    # ── CPU pinning (BUG 4 fix: dùng total_workers thực, không phải config) ─
    if total_workers is None:
        total_workers = NUM_GPUS * WORKERS_PER_GPU
    total_cores     = psutil.cpu_count(logical=True)
    cores_per_worker = max(1, total_cores // total_workers)
    cpu_start = worker_id * cores_per_worker
    try:
        psutil.Process().cpu_affinity(
            list(range(cpu_start, cpu_start + cores_per_worker))
        )
    except Exception:
        pass

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # PyTorch performance flags
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    from detectors.retinaface.detector import LandmarksDetector
    from detectors.retinaface.video_process import VideoProcess
    from fast_pipeline import BatchedLandmarkPipeline

    landmarks_obj = LandmarksDetector(device='cuda:0')
    video_process = VideoProcess(convert_gray=False)
    tokenizer     = TextTransform()

    # Retry pass dùng chunk nhỏ hơn nhiều — đó là các video đã OOM ở pass chính
    det_chunk = RETRY_DET_CHUNK if retry_mode else DET_CHUNK
    fan_chunk = RETRY_FAN_CHUNK if retry_mode else FAN_CHUNK

    fast_pipeline = BatchedLandmarkPipeline(
        landmarks_obj.face_detector,
        landmarks_obj.landmark_detector,
        device='cuda:0',
        det_chunk=det_chunk,
        fan_chunk=fan_chunk,
        fp16=USE_FP16,
        use_compile=USE_COMPILE,
    )

    log_suffix = f"w{worker_id}.retry" if retry_mode else f"w{worker_id}"
    train_csv = open(f"{LABELS_DIR}/train_{log_suffix}.csv", "a")
    val_csv   = open(f"{LABELS_DIR}/val_{log_suffix}.csv",   "a")
    test_csv  = open(f"{LABELS_DIR}/test_{log_suffix}.csv",  "a")
    done_log  = open(f"{DONE_LOG}.{log_suffix}", "a")
    # OOM-deferred log: ở pass chính, các video không retry được sẽ vào đây để
    # pass cuối (single-worker) chạy lại. Ở pass retry, file này KHÔNG được mở
    # tránh đệ quy vô hạn — OOM ở pass retry sẽ ghi vào done_log như INFER-ERR.
    oom_log   = open(f"{OOM_LOG}.{log_suffix}", "a") if not retry_mode else None

    # ── BUG 1 fix: thread-safe file writes ─────────────────────────────────
    # CSV rows can exceed PIPE_BUF (4096B) when `ids` is long → 3 save_consumer
    # threads writing concurrently can interleave bytes mid-row, corrupting
    # train/val/test.csv. done_log/oom_log writes are short but written by 8+
    # threads (GPU loop + crop_consumer×6 + save_consumer×3) — safer with lock.
    csv_lock = threading.Lock()
    log_lock = threading.Lock()

    def _write_done(name):
        with log_lock:
            done_log.write(name + '\n'); done_log.flush()

    def _write_oom(path):
        with log_lock:
            oom_log.write(path + '\n'); oom_log.flush()

    def _write_csv(handle, row):
        with csv_lock:
            handle.write(row); handle.flush()

    total_files  = counters['total']
    initial_done = counters['initial_done']

    # ── Stage 1: Decode  ───────────────────────────────────────────────────
    decode_q = queue.Queue(maxsize=PREFETCH_Q)

    def decode_one(f):
        # BUG 8 fix: outer try/except guarantees decode_q ALWAYS receives an
        # entry per file — without it, a freak error in Path(f).name or in
        # decode_q.put would leave `pending` counter unsatisfied and the main
        # GPU loop would hang forever on decode_q.get().
        try:
            name = Path(f).name
            if name in done_set:
                decode_q.put((f, 'resume', None))
                return
            try:
                decode_q.put((f, 'ok', load_video(f)))
            except Exception:
                decode_q.put((f, 'error', None))
        except Exception:
            try:
                decode_q.put((f, 'error', None))
            except Exception:
                pass

    def decode_producer():
        with ThreadPoolExecutor(max_workers=DECODE_THREADS) as pool:
            for fut in [pool.submit(decode_one, f) for f in file_chunk]:
                fut.result()

    decode_thread = threading.Thread(target=decode_producer, daemon=True)
    decode_thread.start()

    # ── Stage 3: Crop  ────────────────────────────────────────────────────
    # KEY: GPU inference puts (frames, landmarks, meta) here;
    # crop workers pick it up so GPU can immediately start the next video.
    crop_q = queue.Queue(maxsize=CROP_Q_DEPTH)

    # ── Stage 4: Save  ────────────────────────────────────────────────────
    save_q = queue.Queue(maxsize=SAVE_Q_DEPTH)

    def crop_consumer():
        while True:
            item = crop_q.get()
            if item is _SENTINEL:
                break
            f         = item['f']
            frames    = item['frames']
            frames_gpu = item['frames_gpu']
            landmarks = item['landmarks']
            idx       = item['idx']
            name = Path(f).name

            elapsed = time.time() - start_time
            ts = time.strftime('%H:%M:%S', time.gmtime(elapsed))

            try:
                video_data = video_process(frames, landmarks, frames_gpu=frames_gpu)
            except (OverflowError, TypeError, AssertionError, UnboundLocalError):
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - CROP-ERR    {name}", flush=True)
                _write_done(name)
                continue

            if video_data is None:
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - NO-FACE     {name}", flush=True)
                _write_done(name)
                continue

            # BUG 5 fix: with_suffix() chỉ thay extension cuối, an toàn với path
            # có chuỗi '.mp4' ở giữa (vd. /archive.mp4_v2/clip.mp4).
            csv_path = str(Path(f).with_suffix('.csv'))
            try:
                transcript = []
                with open(csv_path) as csvfile:
                    for row in csv.DictReader(csvfile):
                        transcript.append(row['Word'].strip().rstrip('.,!?\'"').lower())
            except FileNotFoundError:
                # BUG 3 fix: trước đây file bị drop âm thầm; thêm log NO-CSV
                # để user biết vì sao thiếu — nhất quán với các nhánh lỗi khác.
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - NO-CSV      {name}", flush=True)
                _write_done(name)
                continue

            text = ' '.join(transcript)
            ids  = ' '.join(str(x) for x in tokenizer.tokenize(text).tolist())
            save_q.put({
                'f': f, 'name': name,
                'video_tensor': torch.tensor(video_data),
                'text': text, 'ids': ids, 'idx': idx,
            })

    def save_consumer():
        while True:
            item = save_q.get()
            if item is _SENTINEL:
                break
            f, name = item['f'], item['name']
            video_tensor, text, ids, idx = (
                item['video_tensor'], item['text'], item['ids'], item['idx'])

            out_vid = f"{OUTPUT_DIR}/video/{name}"
            out_txt = f"{OUTPUT_DIR}/text/{Path(f).stem}.txt"
            vid_np  = video_tensor.numpy()

            save2vid(out_vid, vid_np, 25)

            os.makedirs(os.path.dirname(out_txt), exist_ok=True)
            with open(out_txt, 'w') as ft:
                ft.write(text)

            speaker = Path(f).stem.rsplit('_', 1)[0]
            row     = f"vnlr,video/{name},{len(video_tensor)},{ids}\n"
            # BUG 1 fix: dùng _write_csv để lock — tránh interleave dòng dài
            # khi 3 save_consumer thread cùng ghi train/val/test.csv.
            if speaker in train_set:
                _write_csv(train_csv, row)
            elif speaker in val_set:
                _write_csv(val_csv, row)
            elif speaker in test_set:
                _write_csv(test_csv, row)

            # BUG 6 lưu ý: ghi done_log SAU CSV để nếu crash giữa 2 lệnh,
            # dedup trong merge_shards sẽ loại trùng (xem _dedup_csv).
            _write_done(name)
            # BUG 7 fix: idx ở GPU loop là "dispatch idx", có thể vượt xa số file
            # đã thực sự lưu. Dùng counter 'saved' riêng cho thông báo OK.
            with lock:
                counters['saved'] = counters.get('saved', 0) + 1
                saved_idx = counters['saved']
            elapsed = time.time() - start_time
            ts = time.strftime('%H:%M:%S', time.gmtime(elapsed))
            print(f"[{ts}] {saved_idx}/{total_files} W{worker_id}/G{gpu_id} - OK   {name}", flush=True)

    crop_threads = [threading.Thread(target=crop_consumer, daemon=True)
                    for _ in range(CROP_WORKERS)]
    save_threads = [threading.Thread(target=save_consumer, daemon=True)
                    for _ in range(SAVE_THREADS)]
    for t in crop_threads + save_threads:
        t.start()

    # ── Stage 2: GPU inference loop (main worker thread) ──────────────────
    # This thread ONLY does GPU inference and immediately queues result for
    # crop workers — it never waits for CPU crop to finish.
    try:
        pending = len(file_chunk)
        while pending > 0:
            f, status, frames = decode_q.get()
            pending -= 1
            name = Path(f).name

            with lock:
                counters['done'] += 1
                idx = counters['done']

            elapsed = time.time() - start_time
            ts = time.strftime('%H:%M:%S', time.gmtime(elapsed))

            if status == 'resume':
                if (idx - initial_done) <= 5 or (idx % 500 == 0):
                    print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - RESUME-SKIP {name}", flush=True)
                continue

            if status == 'error':
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - DECODE-ERR  {name}", flush=True)
                _write_done(name)
                continue

            try:
                landmarks, frames_gpu = _infer_with_oom_retry(fast_pipeline, frames)
            except Exception as _infer_exc:
                # Phân nhánh: OOM (kể cả khi bị TorchScript bọc thành RuntimeError)
                # vs. lỗi inference khác. Trước đây chỉ bắt torch.cuda.OutOfMemoryError
                # khiến OOM trong scripted module bị xem nhầm là INFER-ERR và đánh dấu
                # done luôn — file đó mất hết cơ hội retry.
                if _is_cuda_oom(_infer_exc):
                    # Đã thử OOM_MAX_ATTEMPTS lần với chunk giảm dần mà vẫn OOM.
                    # Defer sang retry pass (1-worker) thay vì đánh dấu done.
                    if oom_log is not None:
                        tag = "OOM-DEFER "
                        _write_oom(f)
                    else:
                        # Đang ở retry pass — không có tầng dự phòng tiếp theo, bỏ luôn.
                        tag = "OOM-FINAL "
                        _write_done(name)
                    print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - {tag} {name}", flush=True)
                    # Giải phóng tham chiếu frames (RAM lớn) và GPU cache trước khi tiếp tục
                    del frames
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue

                # Không phải OOM → INFER-ERR thật, đánh dấu done để khỏi retry vô hạn
                import traceback as _tb
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - INFER-ERR   {name} | {type(_infer_exc).__name__}: {_infer_exc!s:.200}", flush=True)
                _tb.print_exc()
                _write_done(name)
                continue

            # Hand off immediately — GPU loop does NOT wait for crop
            # frames_gpu reuses the tensor already on-device from inference (no second upload)
            crop_q.put({'f': f, 'frames': frames, 'frames_gpu': frames_gpu,
                        'landmarks': landmarks, 'idx': idx})

    finally:
        decode_thread.join()
        for _ in crop_threads:
            crop_q.put(_SENTINEL)
        for t in crop_threads:
            t.join()
        for _ in save_threads:
            save_q.put(_SENTINEL)
        for t in save_threads:
            t.join()
        train_csv.close(); val_csv.close(); test_csv.close(); done_log.close()
        if oom_log is not None:
            oom_log.close()


def _shard_suffixes(num_workers):
    """All shard suffixes a worker may have written (normal + retry pass)."""
    suffixes = [f"w{w}" for w in range(num_workers)]
    suffixes += [f"w{w}.retry" for w in range(num_workers)]
    return suffixes


def _dedup_csv_inplace(path):
    """Loại dòng trùng theo cột 'video/{name}' (cột thứ 2) — giữ entry CUỐI.
    BUG 6 fix: nếu worker chết giữa CSV-write và done_log-write, file sẽ
    được xử lý lại lần sau → CSV xuất hiện 2 row cho cùng video. Hàm này
    consolidate sau merge để dataset cuối cùng sạch.
    """
    if not os.path.exists(path):
        return
    seen = {}
    order = []  # giữ thứ tự xuất hiện đầu tiên cho ổn định
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split(',', 2)
            key = parts[1] if len(parts) >= 2 else line
            if key not in seen:
                order.append(key)
            seen[key] = line  # ghi đè → giữ row mới nhất
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        for key in order:
            f.write(seen[key])
    os.replace(tmp, path)


def _dedup_lines_inplace(path):
    """Loại dòng trùng (so sánh nguyên vẹn). Dùng cho DONE_LOG."""
    if not os.path.exists(path):
        return
    seen = set()
    out_lines = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out_lines.append(line if line.endswith('\n') else line + '\n')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.writelines(out_lines)
    os.replace(tmp, path)


def merge_shards(num_workers):
    """Merge per-worker CSV + done-log shards into the canonical files.
    Bao gồm cả shard từ retry pass (suffix '.retry').
    Sau khi merge, dedup file gộp để loại trùng do crash giữa các stage.
    """
    suffixes = _shard_suffixes(num_workers)
    for split in ('train', 'val', 'test'):
        canonical = f"{LABELS_DIR}/{split}.csv"
        with open(canonical, "a") as out:
            for sfx in suffixes:
                shard = f"{LABELS_DIR}/{split}_{sfx}.csv"
                if os.path.exists(shard):
                    with open(shard) as inp:
                        out.write(inp.read())
                    os.remove(shard)
        _dedup_csv_inplace(canonical)
    with open(DONE_LOG, "a") as out:
        for sfx in suffixes:
            shard = f"{DONE_LOG}.{sfx}"
            if os.path.exists(shard):
                with open(shard) as inp:
                    out.write(inp.read())
                os.remove(shard)
    _dedup_lines_inplace(DONE_LOG)


def collect_oom_files(num_workers):
    """Read & remove per-worker OOM shards. Returns list of full file paths."""
    files = []
    for w in range(num_workers):
        shard = f"{OOM_LOG}.w{w}"
        if os.path.exists(shard):
            with open(shard) as inp:
                files.extend(line.strip() for line in inp if line.strip())
            os.remove(shard)
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f); unique.append(f)
    return unique


def load_done_set():
    """Aggressively load every possible done-log location so resume always works."""
    done_set = set()
    paths_checked = []

    def absorb(path):
        paths_checked.append(path)
        if os.path.exists(path):
            with open(path) as f:
                added = set(line.strip() for line in f if line.strip())
            done_set.update(added)
            return len(added)
        return 0

    main_count = absorb(DONE_LOG)
    print(f"  main done log [{DONE_LOG}]: {main_count} entries")

    # Absorb leftover shards from previous runs (gpu-style, w-style, and retry-style)
    leftover = 0
    for w in range(NUM_GPUS * WORKERS_PER_GPU + NUM_GPUS):  # cover both schemes
        for pat in (f"{DONE_LOG}.w{w}",
                    f"{DONE_LOG}.w{w}.retry",
                    f"{DONE_LOG}.gpu{w}"):
            if os.path.exists(pat):
                leftover += absorb(pat)
    if leftover:
        print(f"  + {leftover} entries from leftover shards")

    return done_set


if __name__ == '__main__':
    # 'spawn' creates clean child processes with no inherited CUDA state.
    # 'fork' (Linux default) copies the parent's file descriptors and can
    # corrupt the CUDA driver context that torch initialises at import time.
    multiprocessing.set_start_method('spawn', force=True)
    start = time.time()

    # Create all required directories upfront
    for dirpath in [
        DATASET_DIR,
        f"{OUTPUT_DIR}/video",
        f"{OUTPUT_DIR}/text",
        LABELS_DIR,
        str(Path(DONE_LOG).parent),
    ]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
            print(f"Created directory: {dirpath}")

    # Probe NVENC once so workers know whether to attempt it.
    # BUG 2 fix: với 'spawn' start_method, gán _NVENC_OK ở __main__ KHÔNG
    # truyền qua worker (mỗi worker re-import module). Phải lưu kết quả probe
    # vào biến local rồi truyền qua worker args.
    import numpy as _np
    _probe = _np.zeros((4, 256, 256, 3), dtype=_np.uint8)
    nvenc_ok = True
    try:
        _write_video_pyav('/tmp/_nvenc_probe.mp4', _probe, 25,
                          'h264_nvenc', {'preset': 'p1', 'gpu': '0'})
        print("NVENC: available — using hardware encoding")
    except Exception as _e:
        nvenc_ok = False
        print(f"NVENC: unavailable ({_e}) — using libx264 ultrafast")
    del _probe

    files = sorted(glob(f'{DATASET_DIR}/*.mp4'))
    print(f"Found {len(files)} files in {DATASET_DIR}")

    print("\nLoading resume state:")
    done_set = load_done_set()
    print(f"  → {len(done_set)} files already processed (will be skipped)\n")

    # Speaker split
    speakers = sorted(set(Path(f).stem.rsplit('_', 1)[0] for f in files))
    train_sp, temp  = train_test_split(speakers, test_size=0.2, random_state=42)
    val_sp, test_sp = train_test_split(temp,     test_size=0.5, random_state=42)
    train_set, val_set, test_set = set(train_sp), set(val_sp), set(test_sp)

    actual_gpus  = min(NUM_GPUS, torch.cuda.device_count())
    total_workers = actual_gpus * WORKERS_PER_GPU
    print(f"Launching {total_workers} workers ({actual_gpus} GPUs × {WORKERS_PER_GPU} workers/GPU)")

    # Round-robin shard so each worker gets evenly-mixed file IDs
    chunks = [files[i::total_workers] for i in range(total_workers)]
    print(f"Files per worker: min={min(len(c) for c in chunks)} max={max(len(c) for c in chunks)}\n")

    with Manager() as manager:
        lock     = manager.Lock()
        counters = manager.dict(
            done=len(done_set),          # already-completed files count toward progress
            total=len(files),
            initial_done=len(done_set),  # snapshot for filtering log spam
            saved=len(done_set),         # BUG 7 fix: completed-save counter, riêng với 'done' (dispatch)
        )

        args = []
        for wid in range(total_workers):
            gpu_id = wid % actual_gpus  # round-robin GPU assignment
            args.append((
                wid, gpu_id, chunks[wid],
                train_set, val_set, test_set,
                done_set, lock, counters, start,
                False,           # retry_mode
                total_workers,   # BUG 4 fix: actual count, not config
                nvenc_ok,        # BUG 2 fix: propagate probe result to spawn workers
            ))

        with Pool(processes=total_workers) as pool:
            pool.starmap(worker, args)

    # ── OOM retry pass: 1 worker, full VRAM, chunk size rất nhỏ ──────────
    oom_files = collect_oom_files(total_workers)
    if oom_files:
        print(f"\n{'='*60}")
        print(f"OOM retry pass: {len(oom_files)} files deferred from main pass")
        print(f"  → spawning 1 worker with DET_CHUNK={RETRY_DET_CHUNK}, "
              f"FAN_CHUNK={RETRY_FAN_CHUNK} on GPU 0")
        print(f"{'='*60}\n")

        # Reload done_set: main pass đã ghi nhiều file mới
        done_set_retry = load_done_set()

        with Manager() as manager:
            lock_r = manager.Lock()
            counters_r = manager.dict(
                done=0,
                total=len(oom_files),
                initial_done=0,
                saved=0,  # BUG 7
            )
            retry_args = [(
                0, 0, oom_files,
                train_set, val_set, test_set,
                done_set_retry, lock_r, counters_r, time.time(),
                True,        # retry_mode
                1,           # total_workers (retry pass = 1 worker)
                nvenc_ok,    # propagate probe result
            )]
            with Pool(processes=1) as pool:
                pool.starmap(worker, retry_args)
    else:
        print("\nNo OOM-deferred files — retry pass skipped.")

    merge_shards(total_workers)

    elapsed = time.time() - start
    print(f"\nDone in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"Total files: {len(files)}")
    if oom_files:
        print(f"OOM-retry pass processed: {len(oom_files)} files")
