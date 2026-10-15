#!/usr/bin/env bash
# Compact view of every logged epoch row (the jsonl is fsynced per epoch, so it
# is the authoritative progress record; the trainer's stdout is block-buffered).
set -u
/root/miniconda3/bin/python - <<'PYEOF'
import json, pathlib
path = pathlib.Path('/root/autodl-tmp/loweye/crop_pest/crop_pest_epochs.jsonl')
rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
print(f"rows={len(rows)}")
for row in rows:
    val = row['val']
    overall = val['operating_point']['overall']
    counts = val['operating_point']['per_class']
    evaluable = {n: v for n, v in val['per_class'].items() if v['map'] is not None}
    ranked = sorted(evaluable.items(), key=lambda kv: -kv[1]['map'])
    print(json.dumps({
        'epoch': row['epoch'],
        'loss': round(row['loss'], 4),
        'loss_components': {k: round(v, 4) for k, v in row['loss_components'].items()},
        'map': row['map'], 'map50': row['map50'],
        'seconds': round(row['seconds'], 1),
        'lr': row['lr'],
        'samples_seen': row['samples_seen'],
        'peak_mem_mb': round(row['peak_mem_bytes'] / 2 ** 20, 1),
        'evaluable_classes': len(evaluable),
        'classes_with_gt': sum(1 for v in counts.values() if v['gt']),
        'operating_point': {k: (round(v, 5) if isinstance(v, float) else v) for k, v in overall.items()},
        'top5_classes_by_map': [(n, round(v['map'], 4)) for n, v in ranked[:5]],
    }, ensure_ascii=False))
PYEOF
echo "=== epoch timing so far ==="
/root/miniconda3/bin/python - <<'PYEOF'
import json, pathlib
path = pathlib.Path('/root/autodl-tmp/loweye/crop_pest/crop_pest_epochs.jsonl')
rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
if rows:
    per = [r['seconds'] for r in rows]
    mean = sum(per) / len(per)
    print(f"epochs={len(rows)} mean_seconds={mean:.1f} total_seconds={sum(per):.1f}")
    print(f"projected_24_epochs_hours={mean * 24 / 3600:.2f}")
PYEOF
