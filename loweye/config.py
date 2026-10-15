"""配置加载。

单一事实来源：config/app.yaml。代码中不得硬编码阈值，一律走这里，
以便方案书里写的每一个数字都能在配置里找到对应项（可复现）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "app.yaml"
DEFAULT_ROUTES_PATH = PROJECT_ROOT / "config" / "routes.yaml"


class ConfigError(RuntimeError):
    """配置缺失或非法。"""


class Config:
    """点号取值的配置对象，例如 cfg.get("retrieval.rrf_k")。"""

    def __init__(self, data: dict[str, Any], root: Path | None = None) -> None:
        self._data = data
        self.root = Path(root) if root else PROJECT_ROOT

    # ---- 读取 ----
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, None)
        if value is None:
            raise ConfigError(f"缺少必需配置项：{dotted}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return self._data

    # ---- 路径 ----
    def path(self, dotted: str) -> Path:
        """取配置中的相对路径，解析为项目内绝对路径（相对路径一律相对项目根）。"""
        raw = self.require(f"paths.{dotted}")
        p = Path(str(raw))
        return p if p.is_absolute() else (self.root / p)

    def abs_path(self, value: str | Path) -> Path:
        """把任意配置值当作路径解析为项目内绝对路径。

        用于不放在 paths.* 下的路径项（例如 severity.rules_file、tools.report.template）。
        """
        p = Path(str(value))
        return p if p.is_absolute() else (self.root / p)

    def ensure_dirs(self) -> list[Path]:
        created = []
        for key in ("data_raw", "data_processed", "index_dir", "assets", "uploads", "out"):
            p = self.path(key)
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                created.append(p)
        return created

    # ---- 环境变量 ----
    @staticmethod
    def env(name: str, default: str | None = None) -> str | None:
        value = os.environ.get(name)
        return value if value else default

    def env_for(self, dotted: str, default: str | None = None) -> str | None:
        """按配置里写的环境变量名去取密钥，避免把密钥写进配置文件。"""
        var = self.get(dotted)
        return self.env(var, default) if var else default

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"Config(root={self.root}, keys={list(self._data)})"


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"配置文件不存在：{cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data, root=cfg_path.resolve().parent.parent)


def load_routes(path: str | Path | None = None) -> dict[str, Any]:
    routes_path = Path(path) if path else DEFAULT_ROUTES_PATH
    if not routes_path.exists():
        raise ConfigError(f"分级规则表不存在：{routes_path}")
    with routes_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
