"""Preserve original source and license texts; small files only."""
import base64,json,urllib.request
from pathlib import Path
root=Path('/root/loweye/data')
urls={
 'ip102_classes.txt':'https://api.github.com/repos/xpwu95/IP102/contents/classes.txt?ref=master',
 'ip102_official_README.md':'https://api.github.com/repos/xpwu95/IP102/contents/README.md?ref=master',
 'dfire_LICENSE.txt':'https://api.github.com/repos/gaia-solutions-on-demand/DFireDataset/contents/LICENSE?ref=master',
}
for name,url in urls.items():
    req=urllib.request.Request(url,headers={'User-Agent':'loweye-academic-source-audit'})
    with urllib.request.urlopen(req,timeout=60) as r: obj=json.load(r)
    raw=base64.b64decode(obj['content']); (root/name).write_bytes(raw)
    print(name,len(raw),obj.get('sha'),flush=True)
