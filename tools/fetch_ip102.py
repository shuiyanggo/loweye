#!/usr/bin/env python3
"""IP102 detection subset, data only; no YOLO code.
Official: https://github.com/xpwu95/IP102 (CVPR2019, Wu et al.)
Official terms: 'This dataset is free for academic usage. For other purposes,
please contact Xiaoping Wu (xpwu95@163.com).'
Mirror: https://www.kaggle.com/datasets/parserpixy/ip102-dataset version1
Mirror PDDL claim does NOT supersede original academic-only terms.
Use only academic local training/evaluation; do not redistribute raw dataset;
commercial deployment and competition/model redistribution permission not established.
Subset claims 18,981 annotated images /102 classes. Archive ~0.81GB;
reserve6GiB for ZIP+extraction+model/cache. No aerial or plant-disease claim.
"""
import argparse, hashlib, json, shutil, time, urllib.request, zipfile
from pathlib import Path
URL='https://www.kaggle.com/api/v1/datasets/download/parserpixy/ip102-dataset?datasetVersionNumber=1'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='/root/loweye/data/external/ip102'); a=ap.parse_args()
    root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free<6*1024**3: raise RuntimeError('Need6GiB free')
    archive=root/'ip102_detection_v1.zip'; partial=archive.with_suffix('.part')
    if not archive.exists():
        from download_transport import download
        download(URL, archive, workers=16)
        if not zipfile.is_zipfile(archive): raise RuntimeError('Downloaded archive is not ZIP')
    with zipfile.ZipFile(archive) as z:
        total=sum(i.file_size for i in z.infolist()); print('zip_bytes',archive.stat().st_size,'extract_bytes',total,'entries',len(z.infolist()),flush=True)
        if shutil.disk_usage(root).free<total+2*1024**3: raise RuntimeError('Insufficient extraction space')
        for m in z.infolist():
            target=(root/m.filename).resolve()
            if not target.is_relative_to(root.resolve()): raise ValueError('Unsafe archive path')
            if not target.exists() or target.stat().st_size!=m.file_size: z.extract(m,root)
    sha=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): sha.update(b)
    rec={'dataset':'IP102 detection subset','source':'https://github.com/xpwu95/IP102','mirror':URL,'original_terms':'free for academic usage; contact author for other purposes','mirror_license':'PDDL claim not relied upon','raw_distribution':'not allowed in this project','commercial_and_competition_distribution':'not established','archive_bytes':archive.stat().st_size,'sha256':sha.hexdigest(),'aerial_validation':'not established','plant_disease':'not represented'}
    (root/'source_license.json').write_text(json.dumps(rec,indent=2),encoding='utf-8'); print(json.dumps(rec),flush=True)
if __name__=='__main__': main()
