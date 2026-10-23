from __future__ import annotations

import argparse
import functools
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "frontend" / "dist"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4173)
    args = ap.parse_args(argv)

    if not (DIST_DIR / "index.html").exists():
        print(f"REFUSING TO START: {DIST_DIR} has no built dashboard (index.html not found).", file=sys.stderr)
        print("Build it first:  cd frontend && npm run build", file=sys.stderr)
        return 2

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(DIST_DIR))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving {DIST_DIR} at http://{args.host}:{args.port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
