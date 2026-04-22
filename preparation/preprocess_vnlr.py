from data.data_module import AVSRDataLoader
from utils import save_vid_aud_txt
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
import csv
from transforms import TextTransform
import time


start = time.time()


video_loader = AVSRDataLoader(modality = "video", detector = "retinaface", convert_gray= False, gpu_type= "cuda")
tokenizer = TextTransform()

files = glob('/home/zap/Downloads/auto_avsr/dataset/*.mp4')
print(f"Found {len(files)} files \n")

#Test run with 1 speaker
# files = files[:10]

print(f"Working with {len(files)} files \n")

#Find all unique speakers
speakers = sorted(set(Path(f).stem.rsplit('_', 1)[0] for f in files))


#Split dataset sing scikit learn because it look more professional even thought I don't know if it's good or not
train, temp = train_test_split(speakers, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(f"Train list: {train}\n")
print(f"Val list: {val}\n")
print(f"Test list: {test}\n")

train_list = []
val_list = []
test_list = []

for i, f in enumerate(files):
    try:
        video_data = video_loader.load_data(f)
    except (OverflowError, TypeError, AssertionError, UnboundLocalError):
        print(f"[{time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}] {i+1}/{len(files)} - SKIPPED {Path(f).name}")
        continue
    if video_data is None:
        continue

    #Read transcript
    transcript = []
    csv_path = f.replace('.mp4', '.csv')
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            words = row['Word']
            transcript.append(words.strip().rstrip('.,!?\'"').lower())
    text = ' '.join(transcript)
    ids = " ".join(str(x) for x in tokenizer.tokenize(text).tolist())

    #If video is in train list, write into train
    if Path(f).stem.rsplit('_', 1)[0] in train:
        train_list.append(Path(f).name + "," + str(len(video_data)) + "," + ids)

    #If video is in val list, write into val
    if Path(f).stem.rsplit('_', 1)[0] in val:
        val_list.append(Path(f).name + "," + str(len(video_data)) + "," + ids)

    #If video is in test list, write into test
    if Path(f).stem.rsplit('_', 1)[0] in test:
        test_list.append(Path(f).name + "," + str(len(video_data)) + "," + ids)

    output_video_path = f"/home/zap/Downloads/auto_avsr/vnlr/video/{Path(f).name}"
    output_text_path = f"/home/zap/Downloads/auto_avsr/vnlr/text/{Path(f).stem}.txt"

    save_vid_aud_txt(output_video_path, None, output_text_path, video_data, None, text, video_fps=25)

    elapsed = time.time() - start
    print(f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed))}] {i+1}/{len(files)} - {Path(f).name}")

with open("/home/zap/Downloads/auto_avsr/labels/train.csv", "w", newline="") as f:
    for name in train_list:
        f.write("vnlr,video/" + name + '\n')

with open("/home/zap/Downloads/auto_avsr/labels/val.csv", "w", newline="") as f:
    for name in val_list:
        f.write("vnlr,video/" + name + '\n')

with open("/home/zap/Downloads/auto_avsr/labels/test.csv", "w", newline="") as f:
    for name in test_list:
        f.write("vnlr,video/" + name + '\n')

elapsed = time.time() - start
print(f"\nDone in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
print(f"Processed: {len(train_list) + len(val_list) + len(test_list)}/{len(files)} files")
print(f"Train: {len(train_list)} | Val: {len(val_list)} | Test: {len(test_list)}")

