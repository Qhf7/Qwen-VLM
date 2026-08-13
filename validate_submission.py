from pathlib import Path
import argparse
from PIL import Image
import csv
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/submission_vit_baseline'))
ap.add_argument('--zip', type=Path, default=None)
args = ap.parse_args()
root = args.root
with (root / 'submission.csv').open(encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
masks = list((root / 'predicted_masks').glob('*/*/*_mask.png'))
assert len(rows) == 750, len(rows)
assert len(masks) == 3750, len(masks)
for p in masks:
    with Image.open(p) as im:
        assert im.mode == 'L', (str(p), im.mode)
        assert im.size == (448, 448), (str(p), im.size)
out = args.zip or (root.parent / f'{root.name}.zip')
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(root / 'submission.csv', 'submission.csv')
    for p in masks:
        z.write(p, p.relative_to(root).as_posix())
print('rows', len(rows), 'masks', len(masks), 'zip_bytes', out.stat().st_size)
print('first_row', rows[0])
print('first_mask', masks[0])
