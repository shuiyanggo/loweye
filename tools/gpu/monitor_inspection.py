"""Bounded background monitor and verified artifact recovery via direct Paramiko.
Credentials and endpoint come from the environment only. Never shuts the paid host
down automatically; never writes to the remote host.
"""
import hashlib,json,os,time
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parents[2]
REMOTE='/root/autodl-tmp/loweye'
# 端点在本会话内换过三次（bjb1:23837 → bjb1:47891 → bjb2:26387）。写死的默认值在主机换址后会
# 静默轮询一个不存在的地址：进程还活着、日志还在打印、回收收据永远不写。所以不留默认值，由
# endpoint_from_env() 在连接前立刻退出（与 tools/remote_gpu.py 对 LOWEYE_GPU_PASS 的处理一致，
# 也避免把「导入即需要凭据」塞给单元测试）。用法见 README §7.8。
HOST=os.environ.get('LOWEYE_GPU_HOST')
PORT=os.environ.get('LOWEYE_GPU_PORT')
def endpoint_from_env():
    """返回 (host,port)。缺环境变量就直接退出，绝不退回某个写死的旧地址。"""
    if not HOST or not PORT:
        raise SystemExit('缺少环境变量 LOWEYE_GPU_HOST / LOWEYE_GPU_PORT（AutoDL 地址与端口，见 README §7.8）')
    if not os.environ.get('LOWEYE_GPU_PASS'):
        raise SystemExit('缺少环境变量 LOWEYE_GPU_PASS（AutoDL root 密码）')
    return HOST,int(PORT)
OUT=ROOT/'out'/'inspection_gpu'; OUT.mkdir(parents=True,exist_ok=True)
TASKS=[('fire_smoke','dfire',640),('crop_pest','ip102',800)]


class _DigestWriter:
    """getfo() 只接受一个可写对象；用它落盘的同时算 SHA256，不再为算摘要读第二遍。"""
    def __init__(self,fh):
        self.fh=fh; self.digest=hashlib.sha256(); self.bytes=0
    def write(self,data):
        self.bytes+=len(data); self.digest.update(data); return self.fh.write(data)
    def flush(self): return self.fh.flush()


def fetch(sftp,remote,local,stall_timeout=180,concurrency=64):
    """下整份远端文件，边落盘边算 SHA256，返回本文件的摘要。

    两点是实测逼出来的，别改回去：
    1) 用 sftp.getfo(prefetch=True) 而不是自己 read(1MB) 循环——后者每次只发一个
       32 KB 的 SSH_FXP_READ 请求，在跨境链路上被往返时延卡住（实测 130 MB 要数分钟），
       prefetch 会并发预取，同一链路快一到两个数量级。
    2) 通道设静默超时：卡死的会话必须抛 OSError 交给 main() 重连重试，而不是无限期
       挂住唯一能做回收的进程（实测第二个大文件曾挂在 0 字节不再前进）。
    """
    local.parent.mkdir(parents=True,exist_ok=True)
    part=local.with_suffix(local.suffix+'.part')
    channel=sftp.get_channel()
    old_timeout=channel.gettimeout() if channel is not None else None
    try:
        if channel is not None: channel.settimeout(stall_timeout)
        with part.open('wb') as dst:
            writer=_DigestWriter(dst)
            sftp.getfo(remote,writer,prefetch=True,
                       max_concurrent_prefetch_requests=concurrency)
    finally:
        if channel is not None:
            try: channel.settimeout(old_timeout)
            except Exception: pass
    if part.stat().st_size!=sftp.stat(remote).st_size:
        raise RuntimeError('Recovery size mismatch: '+remote)
    part.replace(local)
    return writer.digest.hexdigest()

def text(sftp,path):
    try:
        with sftp.open(path,'rb') as f: return f.read().decode('utf-8')
    except OSError: return ''

def main():
    host,port=endpoint_from_env()
    recovered=set(); reported=set(); deadline=time.monotonic()+72*3600
    while time.monotonic()<deadline:
        client=paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(host,port=port,username='root',password=os.environ['LOWEYE_GPU_PASS'],timeout=30)
            sftp=client.open_sftp()
            report={}
            for task,dataset,size in TASKS:
                raw=text(sftp,f'/root/loweye/logs/{task}_pipeline_status.json')
                state=json.loads(raw) if raw else {'phase':'not_started'}
                report[task]=state
                archive='dfire_v1' if dataset=='dfire' else 'ip102_detection_v1'
                ledger=text(sftp,f'/root/loweye/data/external/{dataset}/{archive}.ranges.json')
                if ledger:
                    value=json.loads(ledger); state['download_blocks']=len(value['completed']); state['download_total_blocks']=(value['identity']['bytes']+2097151)//2097152
                log=text(sftp,f'{REMOTE}/{task}/{task}_epochs.jsonl')
                if log:
                    rows=log.strip().splitlines(); state['latest_epoch']=json.loads(rows[-1])
                    state['epochs_logged']=len(rows)
                    (OUT/f'{task}_epochs.jsonl').write_text(log,encoding='utf-8')
                if state['phase']=='failed' and (task,state.get('error')) not in reported:
                    reported.add((task,state.get('error')))
                    print('TASK_FAILED',task,json.dumps(state,ensure_ascii=False),flush=True)
                    print(text(sftp,f'/root/loweye/logs/train_{task}.log')[-2000:],flush=True)
                if state['phase']=='complete' and task not in recovered:
                    pairs=[(f'{REMOTE}/models/{task}_fcos_{size}.onnx',ROOT/'models'/'detector'/f'{task}_fcos_{size}.onnx'),
                           (f'{REMOTE}/models/{task}_fcos_metadata.json',OUT/f'{task}_fcos_metadata.json'),
                           (f'{REMOTE}/{task}/{task}_fcos_best.pth',ROOT/'models'/'detector'/f'{task}_fcos_best.pth'),
                           (f'{REMOTE}/{task}/{task}_fcos_last.pth',ROOT/'models'/'detector'/f'{task}_fcos_last.pth'),
                           (f'{REMOTE}/{task}/{task}_test.json',OUT/f'{task}_test.json'),
                           (f'{REMOTE}/{task}/{task}_run.json',OUT/f'{task}_run.json'),
                           (f'/root/loweye/data/{task}_manifest.json',OUT/f'{task}_manifest.json')]
                    chosen=state.get('manifest')
                    if chosen and chosen!=f'/root/loweye/data/{task}_manifest.json':
                        pairs.append((chosen,OUT/f'{task}_group_safe_manifest.json'))
                    pairs.append((f'/root/loweye/logs/train_{task}.log',OUT/f'train_{task}.log'))
                    receipts=[]
                    for remote,local in pairs:
                        digest=fetch(sftp,remote,local)
                        _,stdout,stderr=client.exec_command('sha256sum '+remote,timeout=120)
                        expected=stdout.read().decode().split()[0]
                        if digest!=expected: raise RuntimeError('Recovery SHA256 mismatch: '+remote)
                        receipts.append({'remote':remote,'local':str(local),'sha256':digest})
                    (OUT/f'{task}_recovery.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
                    recovered.add(task)
                    print('RECOVERED',task,json.dumps(receipts,ensure_ascii=False),flush=True)
            (OUT/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            concise={}
            for k,v in report.items():
                row={x:v[x] for x in ('phase','download_blocks','download_total_blocks','epochs_logged') if x in v}
                if 'latest_epoch' in v:
                    e=v['latest_epoch']; row['epoch']=e.get('epoch'); row['map']=e.get('map')
                    row['map50']=e.get('map50'); row['loss']=round(e.get('loss',0),4); row['sec']=round(e.get('seconds',0),1)
                concise[k]=row
            print(json.dumps(concise,ensure_ascii=False),flush=True)
            if len(recovered)==2:
                print('Both models recovered with SHA256 verification. AutoDL remains ON; stop billing after verification.',flush=True)
                return
        except (OSError,paramiko.SSHException,RuntimeError) as exc:
            # RuntimeError covers 'Recovery size mismatch' / 'Recovery SHA256 mismatch': a
            # flaky transfer must not kill the only process able to recover the artifacts --
            # it holds the in-memory `recovered` set, so a crash re-downloads gigabytes.
            print('monitor_retry',type(exc).__name__,str(exc)[:200],flush=True)
        finally: client.close()
        time.sleep(120)
    raise RuntimeError('72h monitor deadline reached; inspect paid host')
if __name__=='__main__': main()
