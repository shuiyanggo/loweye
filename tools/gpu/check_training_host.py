import os,subprocess
from pathlib import Path
import torch,torchvision,torchmetrics,onnx,onnxruntime,pycocotools
print('versions',torch.__version__,torchvision.__version__,torchmetrics.__version__,onnxruntime.__version__,'cuda',torch.cuda.is_available(),flush=True)
subprocess.run(['ps','-p','2847,2947','-o','pid,etime,%cpu,args'])
for task in ('dfire','ip102'):
 p=Path('/root/loweye/logs')/f'fetch_{task}.log'
 print(task,p.read_text()[-2500:])
 for f in (Path('/root/loweye/data/external')/task).glob('*.part'): print(f,f.stat().st_size)
subprocess.run(['df','-h','/root/loweye'])
