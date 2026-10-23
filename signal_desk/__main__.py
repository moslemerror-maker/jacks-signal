from __future__ import annotations

import logging
import sys
import webbrowser

from .config import STATE_DIR, Settings


def _arg(name: str) -> str | None:
    if name in sys.argv:
        i = sys.argv.index(name)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    return None


def _run_app() -> int:
    import datetime as dt
    import threading

    from .clock import IST
    from .engine import Engine
    from .server import serve

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(STATE_DIR / "signal_desk.log", encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)
    log = logging.getLogger("signal_desk")
    s = Settings.load()
    url = f"http://127.0.0.1:{s.port}"
    try:
        httpd = serve(None, s.port)
    except OSError:
        msg = f"Signal Desk is already running - open {url}"
        print(msg)
        log.info(msg)
        if "--no-browser" not in sys.argv:
            webbrowser.open(url)
        return 0
    eng = Engine(s)
    httpd.RequestHandlerClass.engine = eng
    eng.start()
    log.info("Signal Desk started on %s", url)
    print(f"\n  Signal Desk running -> {url}\n  Alerts only: it never places orders. Close this window to stop.\n")

    until = _arg("--until")
    if until and dt.datetime.now(IST).time() >= dt.time(*map(int, until.split(":"))):
        log.info("started after %s - auto-stop not applied today (stop it with stop_signal_desk.bat)", until)
        until = None
    if until:
        h, m = map(int, until.split(":"))

        def _auto_stop() -> None:
            while True:
                now = dt.datetime.now(IST)
                if now.time() >= dt.time(h, m) or now.date() != start_day:
                    log.info("auto-stop at %s", until)
                    httpd.shutdown()
                    return
                threading.Event().wait(30)
        start_day = dt.datetime.now(IST).date()
        threading.Thread(target=_auto_stop, daemon=True).start()

    if "--no-browser" not in sys.argv:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()
        eng.save()
        httpd.server_close()
        log.info("Signal Desk stopped")
    return 0


def main() -> int:
    args = sys.argv[1:]
    cmd = args[0] if args and not args[0].startswith("-") else "run"
    if cmd == "check":
        from .selfcheck import main as check
        return check(args)
    if cmd == "research":
        from .research import main as research
        return research(args[1:])
    if cmd == "stock-research":
        from .stock_research import main as stock_research
        return stock_research(args[1:])
    if cmd == "simulate":
        from .sim import run_day
        run_day()
        return 0
    return _run_app()


if __name__ == "__main__":
    sys.exit(main())
