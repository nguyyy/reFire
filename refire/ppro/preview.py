"""Preview the Premiere panel in a browser -- no Premiere, no real runs, no spend.

    uv run python refire/ppro/preview.py              # http://127.0.0.1:8090/
    http://127.0.0.1:8090/?speed=120                  # replay the run 120x (default 30x)
    http://127.0.0.1:8090/?run=fail                   # the story pass dies

Serves this folder and injects preview.js (a fake Premiere + a replayed `refire make`)
into index.html on the way out; the file on disk never references it. The page
reloads itself when anything in this folder changes, so edit and look.
"""
from __future__ import annotations

import http.server
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 8090


def inject(html: str) -> str:
    return html.replace("<head>", '<head>\n<script src="preview.js"></script>', 1)


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")   # live reload must see the edit
        super().end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/__mtime":
            body, kind = str(max(p.stat().st_mtime_ns for p in HERE.rglob("*")
                                 if p.is_file())), "text/plain"
        elif path in ("/", "/index.html"):
            body, kind = inject((HERE / "index.html").read_text(encoding="utf-8")), "text/html"
        else:
            return super().do_GET()
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main(argv: list[str]) -> int:
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else PORT
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port),
                                             partial(Handler, directory=str(HERE)))
    print("reFire panel preview: http://127.0.0.1:%d/  (Ctrl+C to stop)" % port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
