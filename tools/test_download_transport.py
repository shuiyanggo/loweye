"""Offline bounded transport regression tests, no credentials or network."""
import base64, hashlib, importlib.util, io, json, os, tempfile, urllib.request, zipfile
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('download_transport', Path(__file__).with_name('download_transport.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_STORED) as z: z.writestr('sample.bin',os.urandom(3*1024**2))
blob=buf.getvalue(); etag='"'+hashlib.md5(blob).hexdigest()+'"'; calls=[]
class Response:
    def __init__(self,start,end):
        self.status=206; self.url='https://storage.invalid/pinned'; self.headers={'Content-Range':f'bytes {start}-{end}/{len(blob)}','ETag':etag,'x-goog-hash':'md5='+base64.b64encode(hashlib.md5(blob).digest()).decode()}; self.body=io.BytesIO(blob[start:end+1])
    def read(self,n): return self.body.read(n)
    def __enter__(self): return self
    def __exit__(self,*a): pass
def open_fake(req,timeout):
    start,end=map(int,req.get_header('Range').split('=')[1].split('-')); calls.append((start,end)); return Response(start,end)
with tempfile.TemporaryDirectory(prefix='transport-test-',dir=Path(__file__).parent) as td:
    archive=Path(td)/'data.zip'; archive.with_suffix('.part').write_bytes(blob[:m.BLOCK])
    with patch.object(urllib.request,'urlopen',open_fake): rec=m.download('https://api.invalid/v1',archive,workers=3)
    assert archive.read_bytes()==blob
    assert rec['sha256']==hashlib.sha256(blob).hexdigest()
    assert (m.BLOCK,len(blob)-1) in calls and (0,m.BLOCK-1) not in calls
    print('PASS resume reuses complete block; bounded Range; full MD5/SHA256/ZIP verification')
with tempfile.TemporaryDirectory(prefix='transport-test-',dir=Path(__file__).parent) as td:
    archive=Path(td)/'data.zip'; archive.with_suffix('.part').write_bytes(b'wrong prefix')
    with patch.object(urllib.request,'urlopen',open_fake):
        try: m.download('https://api.invalid/v1',archive)
        except RuntimeError as exc: assert 'does not match' in str(exc)
        else: raise AssertionError('mixed source accepted')
    print('PASS mismatched old source refused')
with tempfile.TemporaryDirectory(prefix='transport-test-',dir=Path(__file__).parent) as td:
    archive=Path(td)/'data.zip'; archive.with_suffix('.part').write_bytes(blob[:m.BLOCK]+b'\0'*(len(blob)-m.BLOCK))
    identity={'source':'https://api.invalid/v1','bytes':len(blob),'etag':etag,'block_bytes':m.BLOCK}
    archive.with_suffix('.ranges.json').write_text(json.dumps({'identity':identity,'completed':[0]}))
    calls.clear()
    with patch.object(urllib.request,'urlopen',open_fake): m.download('https://api.invalid/v1',archive,workers=2)
    assert archive.read_bytes()==blob and (0,m.BLOCK-1) not in calls
    print('PASS preallocated partial resumes from durable checkpoint')
with tempfile.TemporaryDirectory(prefix='transport-test-',dir=Path(__file__).parent) as td:
    archive=Path(td)/'data.zip'; archive.with_suffix('.part').write_bytes(blob)
    archive.with_suffix('.ranges.json').write_text(json.dumps({'identity':{'source':'different-version'},'completed':[]}))
    with patch.object(urllib.request,'urlopen',open_fake):
        try: m.download('https://api.invalid/v1',archive)
        except RuntimeError as exc: assert 'identity changed' in str(exc)
        else: raise AssertionError('changed identity accepted')
    print('PASS changed checkpoint version refused')
def ignore_range(req,timeout):
    r=Response(0,65535); r.status=200; return r
with tempfile.TemporaryDirectory(prefix='transport-test-',dir=Path(__file__).parent) as td:
    archive=Path(td)/'data.zip'
    with patch.object(urllib.request,'urlopen',ignore_range):
        try: m.download('https://api.invalid/v1',archive)
        except RuntimeError as exc: assert 'does not honor' in str(exc)
        else: raise AssertionError('unbounded response accepted')
    assert not archive.with_suffix('.part').exists()
    print('PASS server ignoring Range refused before any archive write')
