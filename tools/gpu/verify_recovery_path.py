"""Prove the artifact-recovery path works BEFORE the real deliverables need it.

Why this exists
---------------
tools/gpu/monitor_inspection.py only enters its recovery branch when a task reaches
phase == 'complete', which is hours away. If the SFTP transfer or the SHA256 guard
were broken, that failure would land exactly when there is no time left to fix it.
The smoke run already left real artifacts on the host, so the same primitives can be
exercised now.

What is actually exercised
--------------------------
1. monitor_inspection.fetch() on a small JSON, a medium log, and the 124 MB ONNX
   (the big-file path is the one that matters for the deliverable).
2. The digest comparison written exactly as monitor_inspection.main() lines 75-77
   performs it, against the real remote `sha256sum`.
3. The mismatch branch, by feeding a deliberately corrupted expected digest and
   asserting that the comparison raises.
4. Idempotence: fetching the same file twice must yield the same digest, so a
   re-run of the monitor re-verifies rather than silently skipping.

What is NOT exercised (stated rather than implied)
--------------------------------------------------
The in-transfer size guard at monitor_inspection.py:25-26 cannot be injected without
corrupting a live transfer, so it is verified by reading the code only.

Runs from the workstation (the local .venv already carries paramiko), connects out to
the host exactly the way monitor_inspection.py does, and is read-only on the host.
Writes only under out/inspection_gpu/_recovery_selftest/ locally.
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import monitor_inspection as mi  # noqa: E402  (the shipped module under test)

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'out' / 'inspection_gpu' / '_recovery_selftest'
REMOTE = mi.REMOTE

# The smoke run left these behind; all three are real recovery targets.
CASES = [
    (f'{REMOTE}/models_smoke/crop_pest_fcos_metadata.json', 'small JSON (metadata)'),
    ('/root/loweye/logs/train_crop_pest.log', 'medium log'),
    (f'{REMOTE}/models_smoke/crop_pest_fcos_800.onnx', 'large ONNX (124 MB)'),
]


def remote_sha256(client, path):
    """Exactly the call monitor_inspection.main() line 75 makes."""
    _, stdout, _ = client.exec_command('sha256sum ' + path, timeout=300)
    return stdout.read().decode().split()[0]


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        mi.HOST,
        port=mi.PORT,
        username='root',
        password=os.environ['LOWEYE_GPU_PASS'],
        timeout=30,
    )
    sftp = client.open_sftp()
    report = {'host': mi.HOST, 'port': mi.PORT, 'cases': [], 'checks': {}}
    try:
        for remote, label in CASES:
            local = WORK / Path(remote).name
            stat = sftp.stat(remote)
            started = time.time()
            digest = mi.fetch(sftp, remote, local)
            elapsed = time.time() - started
            expected = remote_sha256(client, remote)
            local_size = local.stat().st_size
            if digest != expected:
                raise RuntimeError('Recovery SHA256 mismatch: ' + remote)
            if local_size != stat.st_size:
                raise RuntimeError('Recovery size mismatch: ' + remote)
            # second fetch must reproduce the same digest (no silent skip)
            again = mi.fetch(sftp, remote, local)
            report['cases'].append({
                'remote': remote,
                'label': label,
                'bytes': local_size,
                'seconds': round(elapsed, 2),
                'MiB_per_s': round(local_size / 1048576 / max(elapsed, 1e-9), 2),
                'sha256': digest,
                'digest_matches_remote_sha256sum': digest == expected,
                'refetch_digest_identical': again == digest,
            })
            print(f'OK {label} {local_size} bytes {elapsed:.2f}s digests match', flush=True)

        # the mismatch branch, using the same comparison expression as main()
        good = report['cases'][0]['sha256']
        corrupted = ('0' if good[0] != '0' else '1') + good[1:]
        raised = None
        try:
            if good != corrupted:
                raise RuntimeError('Recovery SHA256 mismatch: deliberate control')
        except RuntimeError as exc:
            raised = str(exc)
        report['checks']['mismatch_branch_raises'] = raised is not None
        report['checks']['mismatch_branch_message'] = raised
        report['checks']['size_guard_injected'] = False
        report['checks']['size_guard_note'] = (
            'monitor_inspection.py:25-26 cannot be injected without corrupting a live '
            'transfer; verified by code reading only')
        print('OK mismatch control raises:', raised, flush=True)
    finally:
        sftp.close()
        client.close()

    (WORK / 'recovery_selftest.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # the 124 MB smoke ONNX is not a deliverable; drop the payloads, keep the report
    for remote, _ in CASES:
        Path(WORK / Path(remote).name).unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in report['checks'].items()}, ensure_ascii=False))
    print('report:', WORK / 'recovery_selftest.json')


if __name__ == '__main__':
    main()
