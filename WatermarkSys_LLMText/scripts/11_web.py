"""启动 Web 演示系统（W10）。

  python scripts/11_web.py            # http://127.0.0.1:8000
  python scripts/11_web.py --port 8010
"""
from __future__ import annotations

import argparse

from common import CODE_ROOT, ensure_dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("缺少 uvicorn，请先 pip install uvicorn fastapi")
    uvicorn.run("wm.web.app:app", host=args.host, port=args.port,
                reload=args.reload, app_dir=str(CODE_ROOT / "src"))


if __name__ == "__main__":
    main()
