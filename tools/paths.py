"""把「文件在哪」这件事集中到一处。

**为什么需要这个文件**

2026-11-06 之前，11 个脚本各自写死了：

    ROOT = Path(r"D:\桌面\教务RAG")
    ENV_FILE = Path(r"D:\桌面\多轮对话机器人项目\.env")

后果有两个：

1. 别人 clone 下来一个脚本都跑不起来——路径不存在。
2. `.env` 跨项目引用：P2 读的是 **P1 目录**里的密钥文件。
   这意味着 P2 永远无法独立部署，也永远不能交给别人用。

现在所有路径都从本文件推导：

    ROOT     项目根目录（由 __file__ 推出来，不写死）
    ENV_FILE 优先 ROOT/.env；找不到再退回旧的跨项目路径，兼容老环境

**新环境怎么配**

    cp .env.example .env      # 填上自己的 SILICONFLOW_API_KEY
"""

import os
from pathlib import Path

# 项目根目录 = 本文件（tools/paths.py）的上一级
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"

# 旧环境把 .env 放在 P1 项目里。保留这个回退，是为了不打断已有的本机环境；
# 新用户不会有这个目录，会走 ROOT/.env。
_LEGACY_ENV = Path(r"D:\桌面\多轮对话机器人项目\.env")


def env_file() -> Path:
    """密钥文件的位置。优先项目自己的 .env，没有再退回旧位置。"""
    own = ROOT / ".env"
    if own.exists():
        return own
    if _LEGACY_ENV.exists():
        return _LEGACY_ENV
    return own


def read_env(name: str) -> str:
    """读一个环境变量。环境变量优先，其次 .env 文件。

    先看环境变量是为了让 CI / 部署时能用 `export` 覆盖文件里的值，
    不用改文件。
    """
    val = os.getenv(name)
    if val:
        return val

    path = env_file()
    if not path.exists():
        raise RuntimeError(
            f"找不到 {path}\n"
            f"请把 .env.example 复制成 .env，并填上你自己的 key。"
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()

    raise RuntimeError(f"{path} 里没有 {name}，请补上这一行。")
