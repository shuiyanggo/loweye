"""Read source metadata only; no large downloads."""
import urllib.request, json
urls = [
'https://www.kaggle.com/api/v1/datasets/list?search=smoke-fire-detection-yolo',
'https://hf-mirror.com/api/datasets?search=IP102&limit=12',
'https://hf-mirror.com/api/datasets?search=D-Fire&limit=12',
'https://xiaopingwu.cn/assets/projects/ip102/Data_Download/Data_Download.html',
'https://www.kaggle.com/api/v1/datasets/list?search=ip102',
]
for url in urls:
    print('\nURL', url, flush=True)
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=25) as r:
            raw=r.read(150000).decode('utf-8','replace')
        try:
            data=json.loads(raw)
            if isinstance(data,list):
                data=[{k:x.get(k) for k in ('id','ref','title','totalBytes','licenseName','gated')} for x in data]
            print(json.dumps(data,ensure_ascii=False)[:16000], flush=True)
        except ValueError: print(raw[:16000], flush=True)
    except Exception as e: print(type(e).__name__,str(e),flush=True)
