from data.data_module import AVSRDataLoader
from utils import save_vid_aud_txt
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
import csv
import os
from transforms import TextTransform
import time


start = time.time()

DATASET_DIR = '/workspace/dataset'
OUTPUT_DIR = '/workspace/vnlr'
LABELS_DIR = '/workspace/labels'
DONE_LOG = '/workspace/auto_avsr/preparation/preprocess_done.txt'

os.makedirs(f"{OUTPUT_DIR}/video", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/text", exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

video_loader = AVSRDataLoader(modality="video", detector="retinaface", convert_gray=False, gpu_type="cuda")
tokenizer = TextTransform()

files = sorted(glob(f'{DATASET_DIR}/*.mp4'))
print(f"Found {len(files)} files\n")

# Resume: load already-processed filenames
if os.path.exists(DONE_LOG):
    with open(DONE_LOG, "r") as f:
        done = set(line.strip() for line in f if line.strip())
    print(f"Resuming: {len(done)} files already processed, skipping them\n")
else:
    done = set()

# Speaker split — must be consistent across runs, so always computed from full file list
speakers = sorted(set(Path(f).stem.rsplit('_', 1)[0] for f in files))
train, temp = train_test_split(speakers, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)
train_set, val_set, test_set = set(train), set(val), set(test)

# Open CSV label files in append mode so partial runs accumulate correctly
train_csv = open(f"{LABELS_DIR}/train.csv", "a")
val_csv   = open(f"{LABELS_DIR}/val.csv",   "a")
test_csv  = open(f"{LABELS_DIR}/test.csv",  "a")
done_log  = open(DONE_LOG, "a")

try:
    for i, f in enumerate(files):
        name = Path(f).name
        if name in done:
            print(f"[{time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}] {i+1}/{len(files)} - SKIP (already done) {name}")
            continue

        try:
            video_data = video_loader.load_data(f)
        except (OverflowError, TypeError, AssertionError, UnboundLocalError):
            print(f"[{time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}] {i+1}/{len(files)} - SKIPPED {name}")
            done_log.write(name + "\n")
            done_log.flush()
            continue
        if video_data is None:
            done_log.write(name + "\n")
            done_log.flush()
            continue

        # Read transcript
        transcript = []
        csv_path = f.replace('.mp4', '.csv')
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                words = row['Word']
                transcript.append(words.strip().rstrip('.,!?\'"').lower())
        text = ' '.join(transcript)
        ids = " ".join(str(x) for x in tokenizer.tokenize(text).tolist())

        output_video_path = f"{OUTPUT_DIR}/video/{name}"
        output_text_path  = f"{OUTPUT_DIR}/text/{Path(f).stem}.txt"
        save_vid_aud_txt(output_video_path, None, output_text_path, video_data, None, text, video_fps=25)

        # Write to the correct CSV label file immediately
        speaker = Path(f).stem.rsplit('_', 1)[0]
        row_str = f"vnlr,video/{name},{len(video_data)},{ids}\n"
        if speaker in train_set:
            train_csv.write(row_str)
            train_csv.flush()
        elif speaker in val_set:
            val_csv.write(row_str)
            val_csv.flush()
        elif speaker in test_set:
            test_csv.write(row_str)
            test_csv.flush()

        # Mark file as done
        done_log.write(name + "\n")
        done_log.flush()

        elapsed = time.time() - start
        print(f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed))}] {i+1}/{len(files)} - {name}")

finally:
    train_csv.close()
    val_csv.close()
    test_csv.close()
    done_log.close()

elapsed = time.time() - start
total_done = len(done) + sum(1 for _ in open(DONE_LOG))
print(f"\nDone in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
print(f"Processed: {total_done}/{len(files)} files")