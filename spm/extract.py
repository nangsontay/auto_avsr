import csv
import glob
import os
import time
import sys


def is_suspicious(words, max_repeat=8):
    for k in range(1, 5):
        count = 0
        for i in range(k, len(words)):
            if words[i] == words[i - k]:
                count += 1
            else:
                count = 0
            if count >= max_repeat * k:
                return True
    return False

def has_chinese(words):
    for w in words:
        if any('\u4e00' <= c <= '\u9fff' for c in w):
            return True
    return False

def is_vietnamese(words):
    if not words:
        return False
    if has_chinese(words):
        return False
    non_ascii = sum(1 for w in words if not w.isascii())
    return non_ascii / len(words) >= 0.4

def delete_corrupted(csv_path):
    mp4_path = os.path.splitext(csv_path)[0] + '.mp4'
    for path in (csv_path, mp4_path):
        if os.path.exists(path):
            os.remove(path)

def split_long_chain(line):
    parts = []
    while len(line) > 4192:
        mid = len(line) // 2
        left = line.rfind(' ', 0, mid)
        right = line.find(' ', mid)
        if left == -1 and right == -1:
            break  # no spaces at all, can't split
        if left == -1:
            split_pos = right
        elif right == -1:
            split_pos = left
        else:
            split_pos = left if (mid - left) <= (right - mid) else right
        parts.append(line[:split_pos])
        line = line[split_pos + 1:]
    parts.append(line)
    return parts

csv_files = sorted(glob.glob('/home/zap/Downloads/auto_avsr/dataset/**/*.csv', recursive=True))
total_files = len(csv_files)

start_time = time.time()
sentences_written = 0
sentences_skipped = 0
words_processed = 0

def print_progress(file_idx):
    elapsed = time.time() - start_time
    done = file_idx + 1
    pct = done / total_files
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total_files - done) / rate if rate > 0 else 0

    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
    elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    bar_width = 30
    filled = int(bar_width * pct)
    bar = '█' * filled + '░' * (bar_width - filled)

    sys.stdout.write(
        f'\r[{bar}] {pct*100:.1f}% {done}/{total_files} | '
        f'{elapsed_str} elapsed ETA {eta_str} | '
        f'✓{sentences_written} ✗{sentences_skipped} w:{words_processed} \n'
    )
    sys.stdout.flush()

with open('input.txt', 'w', encoding = 'utf-8') as out:
    current = []  # list of (word, file_idx)
    for file_idx, csv_file in enumerate(csv_files):
        #out.write(csv_file + '\n')
        with open(csv_file, newline = '', encoding = 'utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row['Word']
                current.append((word.strip().rstrip('.,!?\'"').lower(), file_idx))
                words_processed += 1
                #End sentence
                if word.endswith('.'):
                    words_only = [w for w, _ in current]
                    if is_suspicious(words_only) or not is_vietnamese(words_only):
                        last_file = current[-1][1]
                        current = [(w, f) for w, f in current if f != last_file]
                        delete_corrupted(csv_files[last_file])
                        sentences_skipped += 1
                    else:
                        line = ' '.join(words_only)
                        if len(line) > 4192:
                            for part in split_long_chain(line):
                                out.write(part + '\n')
                                sentences_written += 1
                        else:
                            out.write(line + '\n')
                            sentences_written += 1
                        current = []
        # flush whatever is left in current at end of each file
        if current:
            words_only = [w for w, _ in current]
            if is_suspicious(words_only) or not is_vietnamese(words_only):
                delete_corrupted(csv_file)
                sentences_skipped += 1
            else:
                line = ' '.join(words_only)
                if len(line) > 4192:
                    for part in split_long_chain(line):
                        out.write(part + '\n')
                        sentences_written += 1
                else:
                    out.write(line + '\n')
                    sentences_written += 1
            current = []
        print_progress(file_idx)

    #Write any leftover
    if current:
        words_only = [w for w, _ in current]
        if not is_suspicious(words_only) and is_vietnamese(words_only):
            line = ' '.join(words_only)
            if len(line) > 4192:
                for part in split_long_chain(line):
                    out.write(part + '\n')
                    sentences_written += 1
            else:
                out.write(line + '\n')
                sentences_written += 1

elapsed = time.time() - start_time
print(f'\nDone in {time.strftime("%H:%M:%S", time.gmtime(elapsed))}')
print(f'Files processed : {total_files}')
print(f'Sentences written: {sentences_written}')
print(f'Sentences skipped: {sentences_skipped}')
print(f'Words processed  : {words_processed}')