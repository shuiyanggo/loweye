"""Relaunch the orchestrator once the in-flight run has dead-ended.

Why this exists
---------------
The coordinator process running right now was started before the D-Fire preparation fix
(--quarantine-invalid, proven necessary by tools/gpu/probe_dfire_prepare.sh: 340
strict-mode label failures). It keeps its command list in memory, so it will still fail
on fire_smoke's preparation and then exit. Without a relaunch fire_smoke would never
start and the paid GPU would sit idle -- the same starvation bug that already left
fire_smoke at waiting_for_verified_download while crop_pest's smoke gate kept failing.

What it does
------------
Waits for exactly that dead end, then starts the fixed orchestrator through
06_launch_pipeline.sh. That launcher refuses to start while its pidfile points at a live
process, so a duplicate coordinator -- two trainers writing the same checkpoints -- is
structurally impossible even if this watcher misfires. It relaunches once, then exits.
It never kills anything and never shuts the host down.
"""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/root/loweye')
LAUNCH = ROOT / 'tools' / 'gpu' / '06_launch_pipeline.sh'


def phase(task):
    path = ROOT / 'logs' / f'{task}_pipeline_status.json'
    if not path.is_file():
        return 'not_started'
    try:
        return json.loads(path.read_text()).get('phase', 'unknown')
    except (OSError, ValueError):
        return 'unreadable'


def decide(crop, fire):
    """Pure decision so the trigger can be tested without a host.

    'launch' only on a genuine dead end: crop_pest finished and fire_smoke has already
    failed its own attempt, or crop_pest itself failed. Anything else -- including a
    healthy run (fire_smoke waiting/training) -- must wait, because relaunching during
    live work would replace the coordinator that is currently driving it.
    """
    if crop == 'complete' and fire == 'failed':
        return 'launch'
    if crop == 'failed':
        return 'launch'
    return 'wait'


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def main():
    deadline = time.time() + 72 * 3600
    launched = False
    while time.time() < deadline:
        crop, fire = phase('crop_pest'), phase('fire_smoke')
        if not launched:
            verdict = decide(crop, fire)
            if verdict == 'wait':
                log(f'waiting: crop_pest={crop} fire_smoke={fire}')
                time.sleep(120)
                continue
            log(f'dead end reached (crop_pest={crop}, fire_smoke={fire}); relaunching')
            done = subprocess.run(['bash', str(LAUNCH)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=300)
            log(f'launcher exit={done.returncode}')
            if done.stdout.strip():
                print(done.stdout.strip(), flush=True)
            if done.stderr.strip():
                print(done.stderr.strip(), flush=True)
            launched = done.returncode == 0 and 'pipeline_pid=' in done.stdout
            if not launched:
                log('launcher did not start a new coordinator; will retry')
            time.sleep(30)
            continue
        if crop != 'failed' and fire != 'failed':
            log(f'fixed orchestrator is driving the pipeline '
                f'(crop_pest={crop}, fire_smoke={fire}); watcher exiting')
            return
        log(f'after relaunch: crop_pest={crop} fire_smoke={fire}')
        time.sleep(120)
    log('watcher deadline reached without confirming progress')


if __name__ == '__main__':
    main()
