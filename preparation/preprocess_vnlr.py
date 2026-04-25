import csv
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from multiprocessing import Lock, Manager, Pool
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from sklearn.model_selection import train_test_split
from torchcodec.decoders import VideoDecoder

from transforms import TextTransform

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = '/app/dataset'
OUTPUT_DIR = '/app/vnlr'
LABELS_DIR = '/app/labels'
DONE_LOG = '/app/preparation/preprocess_done.txt'

NUM_GPUS         = 4
IO_THREADS       = 8   # prefetch/decode threads per worker (Samsung 990 Pro: 6 GB/s, easily feeds 8)
SAVE_THREADS     = 4   # async save threads per worker — writes never block GPU inference
DETECT_BATCH     = 32  # frames per RetinaFace batch call
USE_NVENC        = True
# ──────────────────────────────────────────────────────────────────────────────


def load_video(path):
    return VideoDecoder(path).get_all_frames().data.permute(0, 2, 3, 1).numpy()


def detect_landmarks(landmarks_detector, landmark_detector_fn, video_frames, batch_size=DETECT_BATCH):
    """Run RetinaFace + FAN frame-by-frame but in larger Python batches
    so GPU stays busy. The ibug API is per-frame, but grouping calls
    keeps CPU↔GPU transfers pipelined."""
    landmarks = []
    for i in range(0, len(video_frames), batch_size):
        chunk = video_frames[i:i + batch_size]
        for frame in chunk:
            detected_faces = landmarks_detector(frame, rgb=False)
            face_points, _ = landmark_detector_fn(frame, detected_faces, rgb=True)
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
            '-preset', 'p1',   # fastest NVENC preset
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


def worker(gpu_id, file_chunk, train_set, val_set, test_set, done_set,
           lock, counters, start_time):
    # Pin this process to its own CPU core range (12 cores each for 4 GPUs / 48 cores)
    cores_per_worker = psutil.cpu_count(logical=True) // NUM_GPUS
    cpu_start = gpu_id * cores_per_worker
    cpu_end   = cpu_start + cores_per_worker
    try:
        psutil.Process().cpu_affinity(list(range(cpu_start, cpu_end)))
    except Exception:
        pass  # cpu_affinity may not be available on all systems

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Import GPU-dependent modules inside the worker so each process owns its CUDA context
    from detectors.retinaface.detector import LandmarksDetector
    from detectors.retinaface.video_process import VideoProcess

    landmarks_detector_obj = LandmarksDetector(device='cuda:0')
    video_process = VideoProcess(convert_gray=False)
    tokenizer = TextTransform()

    face_det  = landmarks_detector_obj.face_detector
    face_land = landmarks_detector_obj.landmark_detector

    # Per-GPU shard CSVs — merged by main process after all workers finish
    train_csv = open(f"{LABELS_DIR}/train_gpu{gpu_id}.csv", "a")
    val_csv   = open(f"{LABELS_DIR}/val_gpu{gpu_id}.csv",   "a")
    test_csv  = open(f"{LABELS_DIR}/test_gpu{gpu_id}.csv",  "a")
    done_log  = open(f"{DONE_LOG}.gpu{gpu_id}", "a")

    total_files = counters['total']

    def process_one(f):
        name = Path(f).name
        if name in done_set:
            return None  # already done

        try:
            video_frames = load_video(f)
        except Exception:
            return ('skip', name, None, None, None)

        try:
            landmarks = detect_landmarks(face_det, face_land, video_frames, DETECT_BATCH)
            video_data = video_process(video_frames, landmarks)
        except (OverflowError, TypeError, AssertionError, UnboundLocalError):
            return ('skip', name, None, None, None)

        if video_data is None:
            return ('skip', name, None, None, None)

        # Read transcript
        csv_path = f.replace('.mp4', '.csv')
        transcript = []
        try:
            with open(csv_path) as csvfile:
                for row in csv.DictReader(csvfile):
                    transcript.append(
                        row['Word'].strip().rstrip('.,!?\'"').lower()
                    )
        except FileNotFoundError:
            return ('skip', name, None, None, None)

        text = ' '.join(transcript)
        ids  = ' '.join(str(x) for x in tokenizer.tokenize(text).tolist())
        video_tensor = torch.tensor(video_data)
        return ('ok', name, video_tensor, text, ids)

    def save_result(f, result, idx):
        """Runs in save_pool — all disk writes for one file."""
        name = Path(f).name
        elapsed = time.time() - start_time
        ts = time.strftime('%H:%M:%S', time.gmtime(elapsed))

        if result is None:
            print(f"[{ts}] {idx}/{total_files} GPU{gpu_id} - SKIP (resume) {name}")
            return

        status, name, video_tensor, text, ids = result

        if status == 'skip':
            print(f"[{ts}] {idx}/{total_files} GPU{gpu_id} - SKIPPED {name}")
            done_log.write(name + '\n'); done_log.flush()
            return

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
        print(f"[{ts}] {idx}/{total_files} GPU{gpu_id} - {name}")

    try:
        # io_pool: decode + GPU inference (CPU/GPU bound)
        # save_pool: all disk writes (I/O bound, overlaps with next inference)
        with ThreadPoolExecutor(max_workers=IO_THREADS) as io_pool, \
             ThreadPoolExecutor(max_workers=SAVE_THREADS) as save_pool:

            infer_futures = {io_pool.submit(process_one, f): f for f in file_chunk}
            save_futures  = []

            for future in as_completed(infer_futures):
                f      = infer_futures[future]
                result = future.result()

                with lock:
                    counters['done'] += 1
                    idx = counters['done']

                # Dispatch save immediately — does not block inference of next file
                sf = save_pool.submit(save_result, f, result, idx)
                save_futures.append(sf)

            # Drain any remaining saves before closing file handles
            for sf in save_futures:
                sf.result()

    finally:
        train_csv.close()
        val_csv.close()
        test_csv.close()
        done_log.close()


def merge_shards():
    for split in ('train', 'val', 'test'):
        with open(f"{LABELS_DIR}/{split}.csv", "a") as out:
            for gpu_id in range(NUM_GPUS):
                shard = f"{LABELS_DIR}/{split}_gpu{gpu_id}.csv"
                if os.path.exists(shard):
                    with open(shard) as inp:
                        out.write(inp.read())
                    os.remove(shard)

    # Merge per-GPU done logs into the main done log
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

    # Resume: load already-processed filenames
    done_set = set()
    if os.path.exists(DONE_LOG):
        with open(DONE_LOG) as f:
            done_set = set(line.strip() for line in f if line.strip())
        # Also absorb any leftover per-GPU shards from a previous interrupted run
        for gpu_id in range(NUM_GPUS):
            shard = f"{DONE_LOG}.gpu{gpu_id}"
            if os.path.exists(shard):
                with open(shard) as f:
                    done_set |= set(line.strip() for line in f if line.strip())
        print(f"Resuming: {len(done_set)} files already done, skipping them")

    # Speaker split — always computed from the full list for consistency
    speakers = sorted(set(Path(f).stem.rsplit('_', 1)[0] for f in files))
    train_sp, temp = train_test_split(speakers, test_size=0.2, random_state=42)
    val_sp, test_sp = train_test_split(temp, test_size=0.5, random_state=42)
    train_set = set(train_sp)
    val_set   = set(val_sp)
    test_set  = set(test_sp)

    # Round-robin shard across GPUs (keeps contiguous runs balanced)
    actual_gpus = min(NUM_GPUS, torch.cuda.device_count())
    chunks = [files[i::actual_gpus] for i in range(actual_gpus)]
    print(f"Using {actual_gpus} GPUs — {[len(c) for c in chunks]} files per GPU\n")

    with Manager() as manager:
        lock     = manager.Lock()
        counters = manager.dict(done=0, total=len(files))

        args = [
            (gpu_id, chunks[gpu_id], train_set, val_set, test_set,
             done_set, lock, counters, start)
            for gpu_id in range(actual_gpus)
        ]

        with Pool(processes=actual_gpus) as pool:
            pool.starmap(worker, args)

    merge_shards()

    elapsed = time.time() - start
    print(f"\nDone in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"Total files: {len(files)}")
