#!/bin/bash
# 在 GPU 机器上把 CPLID 全量数据拉下来（hf-mirror 可达，比从本机上传 418 MB 划算）
PY=/root/miniconda3/bin/python3
mkdir -p /root/loweye/data /root/loweye/logs /root/loweye/tools
cd /root/loweye
echo "=== 安装 huggingface_hub（阿里云源）==="
timeout 300 $PY -m pip install -q huggingface_hub 2>&1 | tail -3
$PY -c "import huggingface_hub as h; print('huggingface_hub', h.__version__)"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME=/root/autodl-tmp/hf

if [ -s /root/loweye/tools/fetch_cplid.py ]; then
  echo "=== 后台启动 CPLID 全量下载 ==="
  nohup $PY tools/fetch_cplid.py --limit 999 > logs/cplid_download.log 2>&1 &
  echo "已在后台启动，PID $!"
else
  echo "缺 tools/fetch_cplid.py，先上传再跑"
  exit 2
fi
