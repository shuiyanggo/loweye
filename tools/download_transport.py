#!/usr/bin/env python3
"""Pinned-source resumable Range downloader, Python standard library only.
No dataset substitution, shell execution, or third-party training code.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

BLOCK = 2 * 1024**2

def download(url: str, archive: Path, *, workers: int = 16) -> dict:
    partial = archive.with_suffix('.part')
    ledger = archive.with_suffix('.ranges.json')
    transport = archive.with_suffix('.transport.json')
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept-Encoding': 'identity'}
    # Resolve pinned API URL once; reuse signed storage URL without printing it.
    request = urllib.request.Request(url, headers={**headers, 'Range': 'bytes=0-65535'})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206:
            raise RuntimeError('Source does not honor bounded Range requests; no full duplicate download started')
        match = re.fullmatch(r'bytes 0-65535/(\d+)', response.headers.get('Content-Range', ''))
        if not match:
            raise RuntimeError('Invalid initial Content-Range')
        total = int(match.group(1)); resolved = response.url
        etag = response.headers.get('ETag'); storage_hash = response.headers.get('x-goog-hash')
        prefix = response.read(65537)
        if len(prefix) != 65536:
            raise RuntimeError('Initial Range length mismatch')
    fingerprint = {'source': url, 'bytes': total, 'etag': etag, 'block_bytes': BLOCK}
    completed = set()
    if ledger.exists():
        state = json.loads(ledger.read_text())
        if state['identity'] != fingerprint:
            raise RuntimeError('Saved Range ledger identity changed; refusing to mix archive versions')
        if not partial.exists() or partial.stat().st_size != total:
            raise RuntimeError('Range ledger exists but partial file shape differs')
        completed = set(state['completed'])
    else:
        have = partial.stat().st_size if partial.exists() else 0
        if have > total:
            raise RuntimeError('Existing partial is larger than source')
        if have:
            with partial.open('rb') as f:
                local_prefix = f.read(min(have, 65536))
            if local_prefix != prefix[:len(local_prefix)]:
                raise RuntimeError('Existing partial does not match pinned remote prefix')
        # Keep only confirmed complete blocks from the old sequential downloader.
        completed = set(range(have // BLOCK))
        with partial.open('r+b' if partial.exists() else 'w+b') as f:
            f.truncate(total); f.flush(); os.fsync(f.fileno())
    lock = threading.Lock(); start = time.monotonic(); transferred = 0
    def save():
        temp = ledger.with_suffix('.tmp')
        temp.write_text(json.dumps({'identity': fingerprint, 'completed': sorted(completed)}))
        temp.replace(ledger)
    save()
    print(json.dumps({'range_download_start': fingerprint, 'workers': workers, 'reused_bytes': len(completed)*BLOCK, 'storage_hash': storage_hash}), flush=True)
    def fetch(index):
        nonlocal transferred
        left = index*BLOCK; right = min(total, left+BLOCK)-1
        for attempt in range(6):
            try:
                h = {**headers, 'Range': f'bytes={left}-{right}'}
                if etag: h['If-Match'] = etag
                req = urllib.request.Request(resolved, headers=h)
                with urllib.request.urlopen(req, timeout=120) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {left}-{right}/{total}':
                        raise RuntimeError('Server ignored Range or returned another archive')
                    if etag and response.headers.get('ETag') != etag:
                        raise RuntimeError('ETag changed')
                    data = response.read(right-left+2)
                    if len(data) != right-left+1:
                        raise RuntimeError('Truncated or excessive Range response')
                with partial.open('r+b', buffering=0) as f:
                    f.seek(left); f.write(data); os.fsync(f.fileno())
                with lock:
                    completed.add(index); transferred += len(data); save()
                    elapsed = time.monotonic()-start
                    print(json.dumps({'completed_ranges': len(completed), 'range_count': (total+BLOCK-1)//BLOCK, 'new_bytes': transferred, 'elapsed_seconds': round(elapsed,2), 'mib_per_second': round(transferred/1048576/max(elapsed,.001),3)}), flush=True)
                return
            except Exception as exc:
                # Exceptions may contain signed URL credentials; log type only.
                print(json.dumps({'range_retry': index, 'attempt': attempt+1, 'error_type': type(exc).__name__}), flush=True)
                if attempt == 5: raise RuntimeError(f'Range {index} failed; partial and ledger retained') from None
                time.sleep(min(2**attempt, 15))
    indices = [i for i in range((total+BLOCK-1)//BLOCK) if i not in completed]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch, indices): pass
    if partial.stat().st_size != total:
        raise RuntimeError('Final archive size differs')
    sha = hashlib.sha256(); md5 = hashlib.md5()
    with partial.open('rb') as f:
        for data in iter(lambda: f.read(8*1024**2), b''):
            sha.update(data); md5.update(data)
    # Google Storage ETags for these pinned archives are single-object MD5s.
    if etag and re.fullmatch(r'[0-9a-fA-F]{32}', etag.strip('\"')):
        if md5.hexdigest().lower() != etag.strip('\"').lower():
            raise RuntimeError('Source ETag MD5 mismatch; archive not promoted')
    if storage_hash:
        import base64
        expected = re.search(r'(?:^|[, ])md5=([^, ]+)', storage_hash)
        if expected and base64.b64encode(md5.digest()).decode() != expected.group(1):
            raise RuntimeError('Storage MD5 mismatch; archive not promoted')
    import zipfile
    with zipfile.ZipFile(partial) as z:
        corrupt = z.testzip()
        if corrupt is not None:
            raise RuntimeError('ZIP member CRC verification failed; archive not promoted')
    record = {**fingerprint, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest(), 'storage_hash': storage_hash, 'new_bytes': transferred, 'elapsed_seconds': round(time.monotonic()-start, 3)}
    transport.write_text(json.dumps(record, indent=2))
    partial.replace(archive)
    print(json.dumps({'range_download_complete': record}), flush=True)
    return record
