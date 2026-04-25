import csv
import os
import subprocess
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from multiprocessing import Manager, Pool
from pathlib import Path

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

NUM_GPUS       = 4
DECODE_THREADS = 8    # threads decoding raw mp4 → numpy frames (pure I/O, no GIL issue)
SAVE_THREADS   = 6    # threads writing output files (NVENC + txt + csv)
PREFETCH_Q     = 24   # decoded-but-not-yet-inferred videos buffered in RAM
                      # 257 GB RAM / 4 workers = ~64 GB each; a 25fps 10s video ≈ 60 MB → fits ~1000
DETECT_BATCH   = 64   # frames fed to RetinaFace per GPU call
USE_NVENC      = True
# ──────────────────────────────────────────────────────────────────────────────


def load_video(path):
    return VideoDecoder(path).get_all_frames().data.permute(0, 2, 3, 1).numpy()


def save2vid_nvenc(filename, vid, fps, gpu_id):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    h, w = vid.shape[1], vid.shape[2]
    proc = subprocess.Popen(
        [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'rgb24', '-r', str(fps),
            '-i', 'pipe:0',
            '-vcodec', 'h264_nvenc', '-gpu', str(gpu_id),
            '-preset', 'p1',
            filename,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(vid.tobytes())
    proc.stdin.close()
    proc.wait()


def save2vid_cpu(filename, vid, fps):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    h, w = vid.shape[1], vid.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    for frame in vid:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


_SENTINEL = object()


def worker(gpu_id, file_chunk, train_set, val_set, test_set, done_set,
           lock, counters, start_time):
    # Pin to 12 cores per worker
    cores_per_worker = psutil.cpu_count(logical=True) // NUM_GPUS
    cpu_start = gpu_id * cores_per_worker
    try:
        psutil.Process().cpu_affinity(list(range(cpu_start, cpu_start + cores_per_worker)))
    except Exception:
        pass

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    from detectors.retinaface.detector import LandmarksDetector
    from detectors.retinaface.video_process import VideoProcess

    landmarks_obj = LandmarksDetector(device='cuda:0')
    video_process = VideoProcess(convert_gray=False)
    tokenizer     = TextTransform()

    face_det  = landmarks_obj.face_detector
    face_land = landmarks_obj.landmark_detector

    train_csv = open(f"{LABELS_DIR}/train_gpu{gpu_id}.csv", "a")
    val_csv   = open(f"{LABELS_DIR}/val_gpu{gpu_id}.csv",   "a")
    test_csv  = open(f"{LABELS_DIR}/test_gpu{gpu_id}.csv",  "a")
    done_log  = open(f"{DONE_LOG}.gpu{gpu_id}", "a")

    total_files = counters['total']

    # ── Stage 1: decode queue (filled by DECODE_THREADS) ─────────────────────
    # Items: (f, video_frames_numpy) or (f, None) for errors
    decode_q = queue.Queue(maxsize=PREFETCH_Q)

    def decode_worker(f):
        name = Path(f).name
        if name in done_set:
            decode_q.put((f, 'resume', None))
            return
        try:
            frames = load_video(f)
            decode_q.put((f, 'ok', frames))
        except Exception:
            decode_q.put((f, 'error', None))

    # Fire off all decodes; queue back-pressure keeps RAM bounded
    decode_thread = threading.Thread(
        target=_decode_producer,
        args=(file_chunk, decode_worker),
        daemon=True,
    )
    decode_thread.start()

    # ── Stage 3: save queue (consumed by SAVE_THREADS) ───────────────────────
    # Items: dict with everything needed to write outputs
    save_q = queue.Queue(maxsize=PREFETCH_Q)

    def save_worker():
        while True:
            item = save_q.get()
            if item is _SENTINEL:
                break
            f, name, video_tensor, text, ids, idx = (
                item['f'], item['name'], item['video_tensor'],
                item['text'], item['ids'], item['idx'],
            )
            out_vid = f"{OUTPUT_DIR}/video/{name}"
            out_txt = f"{OUTPUT_DIR}/text/{Path(f).stem}.txt"
            vid_np  = video_tensor.numpy()

            if USE_NVENC:
                try:
                    save2vid_nvenc(out_vid, vid_np, 25, gpu_id)
                except Exception:
                    save2vid_cpu(out_vid, vid_np, 25)
            else:
                save2vid_cpu(out_vid, vid_np, 25)

            os.makedirs(os.path.dirname(out_txt), exist_ok=True)
            with open(out_txt, 'w') as ft:
                ft.write(text)

            speaker = Path(f).stem.rsplit('_', 1)[0]
            row_str = f"vnlr,video/{name},{len(video_tensor)},{ids}\n"
            if speaker in train_set:
                train_csv.write(row_str); train_csv.flush()
            elif speaker in val_set:
                val_csv.write(row_str);   val_csv.flush()
            elif speaker in test_set:
                test_csv.write(row_str);  test_csv.flush()

            done_log.write(name + '\n'); done_log.flush()

            elapsed = time.time() - start_time
            ts = time.strftime('%H:%M:%S', time.gmtime(elapsed))
            print(f"[{ts}] {idx}/{total_files} GPU{gpu_id} - {name}")

    save_threads = [threading.Thread(target=save_worker, daemon=True)
                    for _ in range(SAVE_THREADS)]
    for t in save_threads:
        t.start()

    # ── Stage 2: GPU inference loop (runs on main worker thread, no GIL fight) ─
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
                print(f"[{ts}] {idx}/{total_files} GPU{gpu_id} - SKIP (resume) {name}")
                continue

            if status == 'error':
                print(f"[{ts}] {idx}/{total_files} GPU{gpu_id} - SKIPPED (decode error) {name}")
                done_log.write(name + '\n'); done_log.flush()
                continue

            # GPU inference — runs uncontested on main worker thread
            try:
                landmarks = _detect_landmarks_batched(face_det, face_land, frames, DETECT_BATCH)
                video_data = video_process(frames, landmarks)
            except (OverflowError, TypeError, AssertionError, UnboundLocalError):
                print(f"[{ts}] {idx}/{total_files} GPU{gpu_id} - SKIPPED (inference error) {name}")
                done_log.write(name + '\n'); done_log.flush()
                continue

            if video_data is None:
                print(f"[{ts}] {idx}/{total_files} GPU{gpu_id} - SKIPPED (no face) {name}")
                done_log.write(name + '\n'); done_log.flush()
                continue

            # Read transcript (fast, tiny CSV)
            csv_path = f.replace('.mp4', '.csv')
            try:
                transcript = []
                with open(csv_path) as csvfile:
                    for row in csv.DictReader(csvfile):
                        transcript.append(row['Word'].strip().rstrip('.,!?\'"').lower())
            except FileNotFoundError:
                done_log.write(name + '\n'); done_log.flush()
                continue

            text         = ' '.join(transcript)
            ids          = ' '.join(str(x) for x in tokenizer.tokenize(text).tolist())
            video_tensor = torch.tensor(video_data)

            # Hand off to save pool — inference loop immediately picks next video
            save_q.put({
                'f': f, 'name': name, 'video_tensor': video_tensor,
                'text': text, 'ids': ids, 'idx': idx,
            })

    finally:
        decode_thread.join()
        for _ in save_threads:
            save_q.put(_SENTINEL)
        for t in save_threads:
            t.join()
        train_csv.close(); val_csv.close(); test_csv.close(); done_log.close()


def _decode_producer(file_chunk, decode_fn):
    """Submit all decodes via a thread pool; back-pressure via bounded queue."""
    with ThreadPoolExecutor(max_workers=DECODE_THREADS) as pool:
        futs = [pool.submit(decode_fn, f) for f in file_chunk]
        for fut in futs:
            fut.result()  # propagate exceptions; queue is filled inside decode_fn


def _detect_landmarks_batched(face_det, face_land, video_frames, batch_size):
    landmarks = []
    for i in range(0, len(video_frames), batch_size):
        for frame in video_frames[i:i + batch_size]:
            detected = face_det(frame, rgb=False)
            face_pts, _ = face_land(frame, detected, rgb=True)
            if len(detected) == 0:
                landmarks.append(None)
            else:
                max_id, max_size = 0, 0
                for idx, bbox in enumerate(detected):
                    sz = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                    if sz > max_size:
                        max_id, max_size = idx, sz
                landmarks.append(face_pts[max_id])
    return landmarks


def merge_shards():
    for split in ('train', 'val', 'test'):
        with open(f"{LABELS_DIR}/{split}.csv", "a") as out:
            for gpu_id in range(NUM_GPUS):
                shard = f"{LABELS_DIR}/{split}_gpu{gpu_id}.csv"
                if os.path.exists(shard):
                    with open(shard) as inp:
                        out.write(inp.read())
                    os.remove(shard)
    with open(DONE_LOG, "a") as out:
        for gpu_id in range(NUM_GPUS):
            shard = f"{DONE_LOG}.gpu{gpu_id}"
            if os.path.exists(shard):
                with open(shard) as inp:
                    out.write(inp.read())
                os.remove(shard)


if __name__ == '__main__':
    start = time.time()

    os.makedirs(f"{OUTPUT_DIR}/video", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/text",  exist_ok=True)
    os.makedirs(LABELS_DIR,            exist_ok=True)

    files = sorted(glob(f'{DATASET_DIR}/*.mp4'))
    print(f"Found {len(files)} files")

    done_set = set()
    if os.path.exists(DONE_LOG):
        with open(DONE_LOG) as f:
            done_set = set(line.strip() for line in f if line.strip())
        for gpu_id in range(NUM_GPUS):
            shard = f"{DONE_LOG}.gpu{gpu_id}"
            if os.path.exists(shard):
                with open(shard) as f:
                    done_set |= set(line.strip() for line in f if line.strip())
        print(f"Resuming: {len(done_set)} files already done")

    speakers = sorted(set(Path(f).stem.rsplit('_', 1)[0] for f in files))
    train_sp, temp  = train_test_split(speakers, test_size=0.2, random_state=42)
    val_sp, test_sp = train_test_split(temp,     test_size=0.5, random_state=42)
    train_set, val_set, test_set = set(train_sp), set(val_sp), set(test_sp)

    actual_gpus = min(NUM_GPUS, torch.cuda.device_count())
    chunks = [files[i::actual_gpus] for i in range(actual_gpus)]
    print(f"Using {actual_gpus} GPUs — {[len(c) for c in chunks]} files per GPU\n")

    with Manager() as manager:
        lock     = manager.Lock()
        counters = manager.dict(done=0, total=len(files))

        args = [
            (i, chunks[i], train_set, val_set, test_set,
             done_set, lock, counters, start)
            for i in range(actual_gpus)
        ]
        with Pool(processes=actual_gpus) as pool:
            pool.starmap(worker, args)

    merge_shards()

    elapsed = time.time() - start
    print(f"\nDone in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"Total files: {len(files)}")
