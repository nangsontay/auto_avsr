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
DONE_LOG     = '/app/preparation/preprocess_done.txt'

NUM_GPUS          = 4
WORKERS_PER_GPU   = 2     # 8 procs total (4 × 2) — 4 workers × ~6-8 GB VRAM each OOMs 24 GB GPUs
DECODE_THREADS    = 4     # per worker — mostly I/O wait, ok to oversubscribe
CROP_WORKERS      = 3     # per worker — pipeline stage: GPU runs while these crop previous video
SAVE_THREADS      = 3     # per worker
PREFETCH_Q        = 16    # decoded frames waiting for GPU
CROP_Q_DEPTH      = 8     # (frames, landmarks) pairs waiting for crop workers
SAVE_Q_DEPTH      = 16    # cropped videos waiting to save
DET_CHUNK         = 128   # 4090: 24 GB VRAM, 2 workers → more headroom per worker
FAN_CHUNK         = 256
USE_FP16          = True
USE_COMPILE       = True  # torch.compile RetinaFace + FAN (~20-40% faster inference)
USE_NVENC         = True
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


def worker(worker_id, gpu_id, file_chunk, train_set, val_set, test_set,
           done_set, lock, counters, start_time):
    # ── CPU pinning ─────────────────────────────────────────────────────────
    total_workers   = NUM_GPUS * WORKERS_PER_GPU
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

    fast_pipeline = BatchedLandmarkPipeline(
        landmarks_obj.face_detector,
        landmarks_obj.landmark_detector,
        device='cuda:0',
        det_chunk=DET_CHUNK,
        fan_chunk=FAN_CHUNK,
        fp16=USE_FP16,
        use_compile=USE_COMPILE,
    )

    train_csv = open(f"{LABELS_DIR}/train_w{worker_id}.csv", "a")
    val_csv   = open(f"{LABELS_DIR}/val_w{worker_id}.csv",   "a")
    test_csv  = open(f"{LABELS_DIR}/test_w{worker_id}.csv",  "a")
    done_log  = open(f"{DONE_LOG}.w{worker_id}", "a")

    total_files  = counters['total']
    initial_done = counters['initial_done']

    # ── Stage 1: Decode  ───────────────────────────────────────────────────
    decode_q = queue.Queue(maxsize=PREFETCH_Q)

    def decode_one(f):
        name = Path(f).name
        if name in done_set:
            decode_q.put((f, 'resume', None))
            return
        try:
            decode_q.put((f, 'ok', load_video(f)))
        except Exception:
            decode_q.put((f, 'error', None))

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
                done_log.write(name + '\n'); done_log.flush()
                continue

            if video_data is None:
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - NO-FACE     {name}", flush=True)
                done_log.write(name + '\n'); done_log.flush()
                continue

            csv_path = f.replace('.mp4', '.csv')
            try:
                transcript = []
                with open(csv_path) as csvfile:
                    for row in csv.DictReader(csvfile):
                        transcript.append(row['Word'].strip().rstrip('.,!?\'"').lower())
            except FileNotFoundError:
                done_log.write(name + '\n'); done_log.flush()
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
            if speaker in train_set:
                train_csv.write(row); train_csv.flush()
            elif speaker in val_set:
                val_csv.write(row);   val_csv.flush()
            elif speaker in test_set:
                test_csv.write(row);  test_csv.flush()

            done_log.write(name + '\n'); done_log.flush()
            elapsed = time.time() - start_time
            ts = time.strftime('%H:%M:%S', time.gmtime(elapsed))
            print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - OK   {name}", flush=True)

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
                done_log.write(name + '\n'); done_log.flush()
                continue

            try:
                landmarks, frames_gpu = fast_pipeline(frames)
            except Exception as _infer_exc:
                import traceback as _tb
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - INFER-ERR   {name} | {type(_infer_exc).__name__}: {_infer_exc!s:.200}", flush=True)
                _tb.print_exc()
                done_log.write(name + '\n'); done_log.flush()
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


def merge_shards(num_workers):
    for split in ('train', 'val', 'test'):
        with open(f"{LABELS_DIR}/{split}.csv", "a") as out:
            for w in range(num_workers):
                shard = f"{LABELS_DIR}/{split}_w{w}.csv"
                if os.path.exists(shard):
                    with open(shard) as inp:
                        out.write(inp.read())
                    os.remove(shard)
    with open(DONE_LOG, "a") as out:
        for w in range(num_workers):
            shard = f"{DONE_LOG}.w{w}"
            if os.path.exists(shard):
                with open(shard) as inp:
                    out.write(inp.read())
                os.remove(shard)


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

    # Absorb leftover shards from previous runs (both old gpu-style and new w-style)
    leftover = 0
    for w in range(NUM_GPUS * WORKERS_PER_GPU + NUM_GPUS):  # cover both schemes
        for pat in (f"{DONE_LOG}.w{w}", f"{DONE_LOG}.gpu{w}"):
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

    # Probe NVENC once so workers know whether to attempt it
    import numpy as _np
    _probe = _np.zeros((4, 256, 256, 3), dtype=_np.uint8)
    try:
        _write_video_pyav('/tmp/_nvenc_probe.mp4', _probe, 25,
                          'h264_nvenc', {'preset': 'p1', 'gpu': '0'})
        print("NVENC: available — using hardware encoding")
    except Exception as _e:
        _NVENC_OK = False
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
        )

        args = []
        for wid in range(total_workers):
            gpu_id = wid % actual_gpus  # round-robin GPU assignment
            args.append((
                wid, gpu_id, chunks[wid],
                train_set, val_set, test_set,
                done_set, lock, counters, start,
            ))

        with Pool(processes=total_workers) as pool:
            pool.starmap(worker, args)

    merge_shards(total_workers)

    elapsed = time.time() - start
    print(f"\nDone in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"Total files: {len(files)}")
