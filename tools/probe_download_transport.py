#!/usr/bin/env python3
"""Bounded, read-only Kaggle v1 transport probe; never downloads full archives."""
import concurrent.futures, hashlib, json, os, subprocess, time, urllib.parse, urllib.request
URLS = {
    'dfire': 'https://www.kaggle.com/api/v1/datasets/download/sayedgamal99/smoke-fire-detection-yolo?datasetVersionNumber=1',
    'ip102': 'https://www.kaggle.com/api/v1/datasets/download/parserpixy/ip102-dataset?datasetVersionNumber=1',
}
def probe(name, url, direct):
    start = time.monotonic(); count = 0; digest = hashlib.sha256(); result = {'dataset': name, 'direct_no_proxy': direct, 'requested_range': 'bytes=0-262143', 'source': url}
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-262143', 'Accept-Encoding': 'identity'})
        with opener.open(req, timeout=12) as response:
            result.update(status=response.status, host=urllib.parse.urlparse(response.url).hostname, headers={k: response.headers.get(k) for k in ['Content-Length', 'Content-Range', 'Accept-Ranges', 'ETag', 'Last-Modified', 'x-goog-hash']})
            # A server ignoring Range still cannot make us read more than 256KiB.
            while count < 262144 and time.monotonic()-start < 35:
                block=response.read(min(16384, 262144-count))
                if not block: break
                digest.update(block); count += len(block)
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc).split('?')[0]
    elapsed=time.monotonic()-start
    result.update(bytes=count, seconds=round(elapsed,3), mib_per_second=round(count/1048576/max(elapsed,.001),5), prefix_sha256=digest.hexdigest())
    return result
if __name__ == '__main__':
    print(json.dumps({'proxy_variables_present':[k for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy'] if os.getenv(k)]}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(probe,n,u,d) for n,u in URLS.items() for d in [False,True]]
        for future in concurrent.futures.as_completed(futures): print(json.dumps(future.result(),sort_keys=True),flush=True)
