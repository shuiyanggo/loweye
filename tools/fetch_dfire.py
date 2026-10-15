#!/usr/bin/env python3
"""D-Fire downloader, data only (no YOLO code).
Source: https://github.com/gaia-solutions-on-demand/DFireDataset
Official-linked mirror: https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo
Dataset collection license: CC0-1.0; upstream disclaims ownership of original images.
License: https://raw.githubusercontent.com/gaia-solutions-on-demand/DFireDataset/master/LICENSE
Academic local training/evaluation only in this project; raw images not redistributed.
No fireworks labels, no claim of aerial validation. Classes 0 smoke, 1 fire (verify YAML).
Download metadata ~3.12GB; reserve 10GiB before download, verify actual archive footprint.
"""
import argparse, hashlib, json, shutil, time, urllib.request, zipfile
from pathlib import Path
URL='https://www.kaggle.com/api/v1/datasets/download/sayedgamal99/smoke-fire-detection-yolo?datasetVersionNumber=1'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='/root/loweye/data/external/dfire'); a=ap.parse_args()
    root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free < 10*1024**3: raise RuntimeError('Need at least 10GiB free for archive+extraction')
    archive=root/'dfire_v1.zip'; partial=archive.with_suffix('.part')
    if not archive.exists():
        from download_transport import download
        download(URL, archive, workers=16)
        if not zipfile.is_zipfile(archive): raise RuntimeError('Downloaded archive is not ZIP')
    with zipfile.ZipFile(archive) as z:
        total=sum(i.file_size for i in z.infolist())
        print('archive_bytes',archive.stat().st_size,'uncompressed_bytes',total,'entries',len(z.infolist()),flush=True)
        if shutil.disk_usage(root).free < total+2*1024**3: raise RuntimeError('Insufficient extraction space')
        for member in z.infolist():
            target=(root/member.filename).resolve()
            if not target.is_relative_to(root.resolve()): raise ValueError('Unsafe archive path')
            if not target.exists() or target.stat().st_size != member.file_size: z.extract(member,root)
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): digest.update(b)
    record={'dataset':'D-Fire','download_url':URL,'source':'https://github.com/gaia-solutions-on-demand/DFireDataset','license':'CC0-1.0 collection; original-image rights disclaimer','distribution':'No raw-data redistribution','archive_bytes':archive.stat().st_size,'sha256':digest.hexdigest(),'fireworks':'not annotated','aerial_validation':'not established'}
    (root/'source_license.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)
if __name__=='__main__': main()
