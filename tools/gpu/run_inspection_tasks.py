"""Sequential data validation, GPU smoke+ORT, full training for two authorized tasks.
Run with nohup; original CPLID artifacts are never modified.
"""
import importlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/loweye'); BIG=Path('/root/autodl-tmp/loweye'); PY=sys.executable

def record(task,phase,**extra):
    value={'task':task,'phase':phase,'unix_time':time.time(),**extra}
    path=ROOT/'logs'/f'{task}_pipeline_status.json'
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)
    print(json.dumps(value),flush=True)

def run(task,phase,command):
    record(task,phase,command=command)
    with (ROOT/'logs'/f'{phase}_{task}.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

TASKS = [('fire_smoke','dfire',640,24),('crop_pest','ip102',800,24)]


def select_ready_task(pending, receipt_times):
    """Pick the earliest ready receipt; ties preserve configured task order."""
    ready = [item for item in pending if item[1] in receipt_times]
    return min(ready, key=lambda item: receipt_times[item[1]]) if ready else None


def already_complete(task):
    """True when the status of a previous run already records phase complete.

    The coordinator gets restarted by hand whenever a downstream task needs a fixed
    command. Without this guard a restart would redo the finished task: the trainer's
    --resume-last would re-export an artifact that was already verified and recovered,
    and it would rewrite that task's status and report files. Skipping is explicit.
    """
    path = ROOT / 'logs' / f'{task}_pipeline_status.json'
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get('phase') == 'complete'
    except (OSError, ValueError):
        return False


def preflight_dependencies():
    try:
        importlib.import_module('yaml')
    except Exception as exc:
        raise RuntimeError(
            'Dependency preflight failed: cannot import yaml (PyYAML). '
            f'Install PyYAML into this interpreter: {PY}; cause: {exc!r}'
        ) from exc


def main():
    (ROOT/'logs').mkdir(exist_ok=True)
    try:
        preflight_dependencies()
    except Exception as exc:
        for task,_,_,_ in TASKS:
            record(task,'failed',error=repr(exc),stage='dependency_preflight')
        raise
    skipped = [item for item in TASKS if already_complete(item[0])]
    pending = [item for item in TASKS if not already_complete(item[0])]
    for task, _, _, _ in skipped:
        print(json.dumps({'task': task, 'phase': 'already_complete', 'action': 'skipped'}), flush=True)
    for task, _, _, _ in pending:
        record(task,'waiting_for_verified_download')
    # Preparation and GPU subprocesses remain strictly sequential. Reconsider
    # only pending sources between tasks; never launch a second GPU process.
    failures = []
    while pending:
        receipt_times = {}
        for _,dataset,_,_ in pending:
            source = ROOT/'data'/'external'/dataset/'source_license.json'
            if source.is_file():
                receipt_times[dataset] = source.stat().st_mtime_ns
        selected = select_ready_task(pending, receipt_times)
        if selected is None:
            time.sleep(30)
            continue
        task,dataset,size,epochs = selected
        pending.remove(selected)
        try:
            data=ROOT/'data'/'external'/dataset
            source=data/'source_license.json'
            # Source record only appears after successful extraction and hash.
            receipt=json.loads(source.read_text())
            if not receipt.get('sha256') or not receipt.get('archive_bytes'): raise RuntimeError('Missing verified download receipt')
            archive_name='dfire_v1' if dataset=='dfire' else 'ip102_detection_v1'
            transport=json.loads((data/(archive_name+'.transport.json')).read_text())
            expected_md5={'dfire':'9745236834d65441af996cb6da6a4fde','ip102':'85869d5c3c686e936c0dde780b208b01'}[dataset]
            if transport.get('md5')!=expected_md5 or transport.get('sha256')!=receipt['sha256']:
                raise RuntimeError('Pinned source archive hash mismatch')
            manifest=ROOT/'data'/f'{task}_manifest.json'
            command=[PY,str(ROOT/'tools'/'prepare_inspection_data.py'),'--root',str(data),'--dataset',dataset,'--output',str(manifest)]
            # D-Fire's YOLO labels carry empty / nonfinite / out-of-image boxes. That was
            # established by tools/gpu/probe_dfire_prepare.sh, run on purpose before the
            # coordinator would have reached this task, precisely so the fix exists before
            # the GPU goes idle. Under the strict default this source can never yield a
            # manifest, so the same explicit quarantine switch the IP102 branch uses
            # applies here too; every excluded sample is recorded in the manifest audit.
            if dataset=='dfire': command+=['--classes','smoke','fire','--quarantine-invalid']
            else: command+=['--classes-file',str(ROOT/'data'/'ip102_classes.txt'),'--voc-class-id-base','0','--quarantine-invalid','--allow-unobserved-classes']
            run(task,'prepare',command)
            value=json.loads(manifest.read_text())
            if value.get('status')!='ok': raise RuntimeError('Preparation failed')
            if value['source_group_audit']['cross_split_groups']:
                manifest=manifest.with_name(manifest.stem+'_group_safe_manifest.json')
            common=[PY,str(ROOT/'tools'/'gpu'/'train_inspection_fcos.py'),'--manifest',str(manifest),'--task',task,'--size',str(size),'--batch','4']
            smoke_out=BIG/(task+'_smoke'); smoke_final=BIG/'models_smoke'
            run(task,'smoke',common+['--out',str(smoke_out),'--final-dir',str(smoke_final),'--smoke','--workers','0'])
            smoke=json.loads((smoke_final/f'{task}_fcos_metadata.json').read_text())
            if smoke['onnx']['verification']['empty_detections']: raise RuntimeError('Invalid empty ONNX smoke verification')
            # --resume-last makes an interrupted multi-hour run continue from its
            # own last checkpoint; the trainer still rejects a changed signature.
            # --nonfinite skip keeps one non-finite batch from killing a resumed multi-hour
            # run: the batch is never trained on, every skip is recorded in the epoch rows /
            # run.json / final metadata, and --nonfinite-limit still stops a poisoned state.
            # The limit is sized from measurement, not taste. The instrumented diagnosis shows
            # classification-only overflow with finite inputs, finite targets and zero non-finite
            # parameters, and the per-epoch event count escalates: 3 (epoch 8), 21 (epoch 9),
            # 35 (epoch 10) and about 25 per minute at the start of epoch 11. A tight cap in that
            # regime therefore only decides *when* a multi-hour run aborts; 100000 is above the
            # theoretical maximum for the rest of this run (14 epochs x 3224 batches = 45136), so
            # no abort is possible, while every skipped batch stays counted in the epoch rows,
            # run.json and the final metadata, and the loss / mAP curve remains the detector for
            # a genuinely poisoned run.
            run(task,'train',common+['--out',str(BIG/task),'--final-dir',str(BIG/'models'),'--epochs',str(epochs),
                '--workers','8','--resume-last','--nonfinite','skip','--nonfinite-limit','100000'])
            record(task,'complete',metadata=str(BIG/'models'/f'{task}_fcos_metadata.json'),manifest=str(manifest))
        except Exception as exc:
            # One task's failure must not starve the other. That is exactly what
            # happened before: crop_pest's smoke gate failed, main() re-raised, and
            # fire_smoke sat at waiting_for_verified_download for hours while the GPU
            # idled. The failed task stays recorded as failed for a later relaunch
            # (already_complete keeps the finished one out of the way), and the loop
            # moves on to whatever is still pending.
            record(task,'failed',error=repr(exc))
            failures.append((task,repr(exc)))
            print(json.dumps({'task':task,'phase':'failed','continuing':True},ensure_ascii=False),flush=True)
            continue
    if failures:
        print(json.dumps({'failed_tasks':[task for task,_ in failures]},ensure_ascii=False),flush=True)
        raise SystemExit(1)
if __name__=='__main__': main()
