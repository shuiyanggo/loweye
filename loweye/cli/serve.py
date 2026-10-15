"""启动 Web 服务。

用法：
    .venv\\Scripts\\python.exe -m loweye.cli.serve
    .venv\\Scripts\\python.exe -m loweye.cli.serve --port 8848 --no-reload
"""

from __future__ import annotations

import argparse
import os

from ..config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="启动低空之眼 Web 服务")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-reload", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    host = args.host or str(cfg.get("server.host", "127.0.0.1"))
    port = args.port or int(cfg.get("server.port", 8848))

    # 让接口层用同一份配置：不传下去的话，接口会用默认 config/app.yaml 建流水线，
    # 表面上「--config 起了服务」，实际跑的还是另一个检测后端 / 另一套索引。
    if args.config:
        os.environ["LOWEYE_CONFIG"] = str(args.config)
        print(f"配置文件         {args.config}")
    print(f"低空之眼 · 控制台地址 http://{host}:{port}/")
    print(f"接口文档         http://{host}:{port}/docs")
    print("自检接口         POST /api/selfcheck（演示前先跑一次）")

    import uvicorn

    uvicorn.run("loweye.api:app", host=host, port=port, reload=not args.no_reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
