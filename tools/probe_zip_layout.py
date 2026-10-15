#!/usr/bin/env python3
"""Read at most 8MiB per pinned archive to inspect ZIP paths, never full data."""
import collections, json, re, struct, urllib.request, zlib
URLS={
 'dfire':'https://www.kaggle.com/api/v1/datasets/download/sayedgamal99/smoke-fire-detection-yolo?datasetVersionNumber=1',
 'ip102':'https://www.kaggle.com/api/v1/datasets/download/parserpixy/ip102-dataset?datasetVersionNumber=1',
}
LIMIT=8*1024**2
class Ranges:
    def __init__(self,url): self.url=url; self.used=0; self.total=None; self.etag=None
    def get(self,left,right):
        length=right-left+1
        if self.used+length>LIMIT: raise RuntimeError('8MiB per-dataset metadata budget exceeded')
        headers={'User-Agent':'Mozilla/5.0','Accept-Encoding':'identity','Range':f'bytes={left}-{right}'}
        if self.etag: headers['If-Match']=self.etag
        req=urllib.request.Request(self.url,headers=headers)
        with urllib.request.urlopen(req,timeout=60) as r:
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
            if r.status!=206 or not match or tuple(map(int,match.groups()[:2]))!=(left,right): raise RuntimeError('Source ignored metadata Range')
            total=int(match.group(3)); etag=r.headers.get('ETag')
            if self.total is not None and (total!=self.total or etag!=self.etag): raise RuntimeError('Source identity changed')
            self.total=total; self.etag=etag; self.url=r.url
            data=r.read(length)
            if len(data)!=length: raise RuntimeError('Range response length mismatch')
            self.used+=length
            return data

def zip64_values(uncompressed, compressed, offset, extra):
    cursor=0; payload=None
    while cursor+4<=len(extra):
        tag,length=struct.unpack_from('<HH',extra,cursor); cursor+=4
        if cursor+length>len(extra): raise RuntimeError('Malformed ZIP extra field')
        if tag==1: payload=extra[cursor:cursor+length]
        cursor+=length
    values=[uncompressed,compressed,offset]; pos=0
    for i,value in enumerate(values):
        if value==0xffffffff:
            if payload is None or pos+8>len(payload): raise RuntimeError('ZIP64 placeholder missing extra value')
            values[i]=struct.unpack_from('<Q',payload,pos)[0]; pos+=8
    return values

def read_member(r, entry):
    name,flags,method,crc,compressed,size,offset=entry
    if flags&1: raise RuntimeError('Encrypted metadata member unsupported')
    if compressed>262144 or size>262144: return {'path':name,'skipped':'member exceeds 256KiB cap','compressed_bytes':compressed,'uncompressed_bytes':size}
    header=r.get(offset,offset+29)
    if header[:4]!=b'PK\x03\x04': raise RuntimeError('Invalid local ZIP header')
    local=struct.unpack('<4s5H3I2H',header)
    start=offset+30+local[-2]+local[-1]
    data=r.get(start,start+compressed-1) if compressed else b''
    if method==0: text=data
    elif method==8:
        dec=zlib.decompressobj(-15); text=dec.decompress(data,262145)
        if len(text)>262144 or not dec.eof: raise RuntimeError('Metadata decompression cap or truncated stream')
    else: return {'path':name,'skipped':'unsupported compression','method':method}
    if len(text)!=size or zlib.crc32(text)&0xffffffff!=crc: raise RuntimeError('Metadata member length/CRC differs')
    decoded=text.decode('utf-8-sig',errors='replace')
    return {'path':name,'uncompressed_bytes':size,'line_count':len(decoded.splitlines()),'content_first12000':decoded[:12000],'content_truncated':len(decoded)>12000}

def inspect(name,url):
    r=Ranges(url); r.get(0,0); total=r.total
    left=max(0,total-65557); tail=r.get(left,total-1)
    pos=len(tail)
    while True:
        pos=tail.rfind(b'PK\x05\x06',0,pos)
        if pos<0: raise RuntimeError('ZIP EOCD missing')
        if pos+22<=len(tail):
            fields=struct.unpack_from('<4s4H2IH',tail,pos)
            if pos+22+fields[-1]==len(tail): break
    _,disk,cd_disk,on_disk,count,size,offset,comment=fields
    if disk or cd_disk or count!=on_disk: raise RuntimeError('Multi-disk ZIP unsupported')
    if count==65535 or size==4294967295 or offset==4294967295: raise RuntimeError('ZIP64 central directory requires separate inspection')
    if offset+size>left+pos: raise RuntimeError('Invalid central directory bounds')
    central=r.get(offset,offset+size-1); cursor=0; names=[]; records=[]; uncompressed=0
    while cursor<len(central):
        if cursor+46>len(central): raise RuntimeError('Truncated central record')
        f=struct.unpack_from('<4s6H3I5H2I',central,cursor)
        if f[0]!=b'PK\x01\x02': raise RuntimeError('Invalid central signature')
        flags=f[3]; filename_len,extra_len,comment_len=f[10:13]
        end=cursor+46+filename_len+extra_len+comment_len
        if end>len(central): raise RuntimeError('Truncated central path')
        namebytes=central[cursor+46:cursor+46+filename_len]
        path=namebytes.decode('utf-8' if flags&2048 else 'cp437'); names.append(path)
        extra=central[cursor+46+filename_len:cursor+46+filename_len+extra_len]
        usize,csize,localoffset=zip64_values(f[9],f[8],f[16],extra)
        records.append((path,flags,f[4],f[7],csize,usize,localoffset))
        uncompressed+=usize; cursor=end
    if len(names)!=count: raise RuntimeError('Central entry count mismatch')
    counts=collections.Counter(); examples=collections.defaultdict(list); suffixes=collections.Counter()
    for path in names:
        if path.endswith('/'): continue
        pieces=path.split('/'); parent='/'.join(pieces[:-1])
        counts[parent]+=1
        if len(examples[parent])<3: examples[parent].append(path)
        suffixes[path.rsplit('.',1)[-1].lower() if '.' in pieces[-1] else '(no suffix)']+=1
    configuration=[n for n in names if n.lower().endswith(('.yaml','.yml','.txt','.csv','.json','.xml')) and not any(x in n.lower() for x in ['/labels/','/annotations/'])]
    selected=[]; first_xml=False
    for record in records:
        path=record[0].lower(); base=path.rsplit('/',1)[-1]
        if name=='dfire' and base in ['data.yaml','data.yml']: selected.append(record)
        if name=='ip102':
            if base in ['test.txt','trainval.txt'] or ('annotations/' in path and path.endswith('.txt')): selected.append(record)
            elif 'annotations/' in path and path.endswith('.xml') and not first_xml:
                selected.append(record); first_xml=True
    metadata=[]
    for record in selected:
        try: metadata.append(read_member(r,record))
        except Exception as exc: metadata.append({'path':record[0],'error_type':type(exc).__name__,'error':str(exc).split('?')[0]})
    print(json.dumps({'metadata_members':metadata,'dataset':name,'source':url,'archive_bytes':total,'etag':r.etag,'metadata_bytes_read':r.used,'entries':count,'uncompressed_bytes':uncompressed,'directory_counts':dict(counts),'directory_examples':dict(examples),'suffix_counts':dict(suffixes),'configuration_or_split_paths_first100':configuration[:100]},ensure_ascii=False),flush=True)
if __name__=='__main__':
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(inspect,n,u):n for n,u in URLS.items()}
        for future in concurrent.futures.as_completed(futures):
            try: future.result()
            except Exception as exc: print(json.dumps({'dataset':futures[future],'error_type':type(exc).__name__,'error':str(exc).split('?')[0]}),flush=True)
