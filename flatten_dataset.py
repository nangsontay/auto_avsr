import os
import glob
import shutil
import sys

DATASET = '/workspace/auto_avsr/dataset'

DRY_RUN = '--run' not in sys.argv  # default is dry-run; pass --run to actually move files

if DRY_RUN:
    print('=== DRY RUN — no files will be moved. Pass --run to execute. ===\n')
else:
    print('=== LIVE RUN — files will be moved. ===\n')

# ── 1. Collect all stems outside the root ────────────────────────────────────
stems = {}  # stem -> folder
for ext in ('mp4', 'csv'):
    for f in glob.glob(os.path.join(DATASET, '**', f'*.{ext}'), recursive=True):
        if os.path.dirname(f) == DATASET:
            continue
        stem = os.path.splitext(os.path.basename(f))[0]
        folder = os.path.dirname(f)
        stems.setdefault(stem, folder)

print(f'Found {len(stems)} unique stems to move.')

# ── 2. Warn about stems missing a paired file ─────────────────────────────────
unpaired = []
for stem, folder in stems.items():
    has_mp4 = os.path.exists(os.path.join(folder, f'{stem}.mp4'))
    has_csv = os.path.exists(os.path.join(folder, f'{stem}.csv'))
    if not (has_mp4 and has_csv):
        unpaired.append((stem, folder, 'mp4' if not has_mp4 else 'csv'))

if unpaired:
    print(f'\nWARNING: {len(unpaired)} stems are missing one file in the pair:')
    for stem, folder, missing in unpaired[:20]:
        print(f'  missing .{missing}: {os.path.join(folder, stem)}')
    if len(unpaired) > 20:
        print(f'  ... and {len(unpaired) - 20} more')
    print()

# ── 3. Build the full move plan ───────────────────────────────────────────────
used_names = set(
    os.path.splitext(f)[0]
    for f in os.listdir(DATASET)
    if f.endswith(('.mp4', '.csv'))
)

plan = []  # list of (src, dst) pairs
plan_dst_stems = set()
errors = []

for stem, folder in stems.items():
    if stem not in used_names and stem not in plan_dst_stems:
        dst_stem = stem
    else:
        parent = os.path.basename(folder)
        dst_stem = f'{parent}_{stem}'
        if dst_stem in used_names or dst_stem in plan_dst_stems:
            grandparent = os.path.basename(os.path.dirname(folder))
            dst_stem = f'{grandparent}_{dst_stem}'
        if dst_stem in used_names or dst_stem in plan_dst_stems:
            errors.append(f'Unresolvable collision for stem: {stem} in {folder}')
            continue

    plan_dst_stems.add(dst_stem)

    for ext in ('mp4', 'csv'):
        src = os.path.join(folder, f'{stem}.{ext}')
        if os.path.exists(src):
            dst = os.path.join(DATASET, f'{dst_stem}.{ext}')
            plan.append((src, dst))

# ── 4. Abort if any planning errors ──────────────────────────────────────────
if errors:
    print(f'\nERROR: {len(errors)} collision(s) could not be resolved. Aborting.')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)

renamed = [(src, dst) for src, dst in plan if os.path.basename(src) != os.path.basename(dst)]
print(f'Plan: {len(plan)} files to move, {len(renamed)} will be renamed to avoid collisions.')
if renamed:
    print('Renamed pairs:')
    for src, dst in renamed[:20]:
        print(f'  {src}  ->  {dst}')
    if len(renamed) > 20:
        print(f'  ... and {len(renamed) - 20} more')

if DRY_RUN:
    print('\nDry run complete. Re-run with --run to execute.')
    sys.exit(0)

# ── 5. Execute with rollback on any error ────────────────────────────────────
completed = []  # (dst, src) for rollback
try:
    for i, (src, dst) in enumerate(plan):
        if os.path.exists(dst):
            raise RuntimeError(f'Destination already exists (should not happen): {dst}')
        shutil.move(src, dst)
        completed.append((dst, src))
        if (i + 1) % 5000 == 0:
            print(f'  moved {i + 1}/{len(plan)} files')

except Exception as e:
    print(f'\nERROR during move: {e}')
    print(f'Rolling back {len(completed)} already-moved files...')
    failed_rollback = []
    for dst, src in reversed(completed):
        try:
            shutil.move(dst, src)
        except Exception as re:
            failed_rollback.append((dst, src, str(re)))
    if failed_rollback:
        print(f'ROLLBACK FAILED for {len(failed_rollback)} files — manual recovery needed:')
        for dst, src, err in failed_rollback:
            print(f'  {dst} -> {src}: {err}')
    else:
        print('Rollback complete. Dataset is unchanged.')
    sys.exit(1)

print(f'\nMoved {len(completed)} files successfully.')

# ── 6. Remove empty subdirectories ───────────────────────────────────────────
removed_dirs = 0
for dirpath, dirnames, filenames in os.walk(DATASET, topdown=False):
    if dirpath == DATASET:
        continue
    if not os.listdir(dirpath):
        os.rmdir(dirpath)
        removed_dirs += 1

print(f'Removed {removed_dirs} empty subdirectories.')
print('Done.')