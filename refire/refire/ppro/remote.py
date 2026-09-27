"""Drive the Premiere panel from a shell -- CEP's devtools port is a CDP endpoint.

`.debug` already puts a Chrome DevTools Protocol server on 8089 whenever the panel
is open (8088 for the AE panel). That is the same protocol devtools itself speaks,
so anything that can talk to it can reach straight through the panel into Premiere's
ExtendScript engine. No panel code, no MCP server, no Premiere CLI (there isn't one).

    python refire/ppro/remote.py 'reFirePpro.probe()'          # extendscript, in premiere
    python refire/ppro/remote.py --panel 'document.title'      # js, in the panel page
    python refire/ppro/remote.py --ae 'app.project.file.fsName'
    python refire/ppro/remote.py --wait 180 'reFirePpro.probe()'   # after launching premiere
    python refire/ppro/remote.py --log '1+1'                   # + panel console output
    python refire/ppro/remote.py --smoke run/<vod>/<run>/ae/manifest.json
    echo <script> | python refire/ppro/remote.py -

Every host call re-evals reFirePpro.jsx first, so edits to the .jsx land without
reopening the panel or restarting Premiere.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from websocket import create_connection

PORT = 8089          # ppro panel; --ae switches to the AE panel's 8088
JSX = Path(__file__).with_name("reFirePpro.jsx")


def _snippet(expr: str, jsx=JSX) -> str:
    """ExtendScript that reloads the jsx, runs `expr`, and can only resolve to a string.

    `expr` must be an expression -- `var x = 1; x` is a *parse* error, which the
    try/catch below cannot catch (nothing compiled) and CEP reports as the useless
    "EvalScript error.". So: code containing `return` is treated as a function body
    and wrapped, which is how you write anything multi-statement.
    """
    if "return" in expr:
        expr = "(function(){%s})()" % expr
    path = str(jsx).replace("\\", "\\\\").replace('"', '\\"')
    return (
        'var __r;try{$.evalFile(new File("%s"));__r=%s;}'
        'catch(e){__r="Error: "+(e.message||e.toString())+(e.line?" [line "+e.line+"]":"");}'
        "String(__r);" % (path, expr)
    )


def _targets(port: int):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/json" % port, timeout=5) as r:
            return [t for t in json.load(r) if t.get("webSocketDebuggerUrl")]
    except (urllib.error.URLError, OSError):
        return []


def wait_for(port: int = PORT, secs: float = 180.0) -> bool:
    """Block until the panel's devtools port answers -- e.g. while Premiere boots."""
    end = time.time() + secs
    while time.time() < end:
        if _targets(port):
            return True
        time.sleep(2)
    return False


def _event(msg: dict) -> str:
    p = msg.get("params", {})
    if msg["method"] == "Runtime.exceptionThrown":
        d = p.get("exceptionDetails", {})
        return "[exception] " + (d.get("exception", {}).get("description") or d.get("text", ""))
    args = [a.get("value", a.get("description", "")) for a in p.get("args", [])]
    return "[%s] %s" % (p.get("type", "log"), " ".join(str(a) for a in args))


def _evaluate(expression: str, port: int, timeout: float, logs: list | None = None):
    targets = _targets(port)
    if not targets:
        sys.exit("no devtools on :%d -- is the reFire panel open in the host app?" % port)

    ws = create_connection(targets[0]["webSocketDebuggerUrl"], timeout=timeout)
    try:
        ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
        ws.send(json.dumps({"id": 2, "method": "Runtime.evaluate", "params": {
            "expression": expression, "awaitPromise": True, "returnByValue": True}}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("method") in ("Runtime.consoleAPICalled", "Runtime.exceptionThrown"):
                if logs is not None:
                    logs.append(_event(msg))
            elif msg.get("id") == 2:
                break
    finally:
        ws.close()

    res = msg.get("result", {})
    if "exceptionDetails" in res:
        return "Error: " + (res["exceptionDetails"].get("exception", {}).get("description")
                            or res["exceptionDetails"].get("text", "eval failed"))
    return res.get("result", {}).get("value")


def call(code, panel=False, port=PORT, timeout=600.0, logs=None, jsx=JSX):
    """Run `code` in the panel page (panel=True) or in the host's ExtendScript engine."""
    if panel:
        return _evaluate(code, port, timeout, logs)
    bridge = "new Promise(function(r){window.__adobe_cep__.evalScript(%s,r);})"
    return _evaluate(bridge % json.dumps(_snippet(code, jsx)), port, timeout, logs)


def smoke(manifest: str, port: int = PORT) -> int:
    """End-to-end check: the jsx loads, a build lands, and the sequence has clips."""
    logs: list = []
    fails = 0

    def check(label, value, ok):
        nonlocal fails
        fails += 0 if ok else 1
        print("%s %s: %s" % ("PASS" if ok else "FAIL", label, value))

    probe = call("reFirePpro.probe()", port=port, logs=logs)
    check("probe", probe, isinstance(probe, str) and probe.startswith("reFirePpro.jsx ok"))
    if fails:  # nothing downstream can pass if the library never reached Premiere
        print("\n".join(logs) or "(no console output)", file=sys.stderr)
        return 1

    before = call("app.project.sequences.numSequences", port=port, logs=logs)
    out = call('reFirePpro.build("%s")' % manifest.replace("\\", "\\\\"),
               port=port, logs=logs)
    check("build", out, isinstance(out, str) and not out.startswith("Error"))
    after = call("app.project.sequences.numSequences", port=port, logs=logs)
    check("new sequence", "%s -> %s" % (before, after), int(after) == int(before) + 1)
    clips = call("app.project.activeSequence.videoTracks[0].clips.numItems",
                 port=port, logs=logs)
    check("v1 clips", clips, int(clips or 0) > 0)
    audio = call("app.project.activeSequence.audioTracks[0].clips.numItems",
                 port=port, logs=logs)
    check("a1 clips", audio, int(audio or 0) > 0)

    if fails:
        print("--- panel console ---\n" + ("\n".join(logs) or "(empty)"), file=sys.stderr)
        pane = call('document.querySelector("#log").textContent.slice(-2000)',
                    panel=True, port=port)
        print("--- panel log ---\n" + (pane or "(empty)"), file=sys.stderr)
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:  # the only fragile bit is backslash escaping into ExtendScript
        assert _snippet("x()").count("\\\\") == str(JSX).count("\\")
        assert json.loads(json.dumps(_snippet("x()")))  # survives the CDP round trip
        print("ok")
        return 0

    port = 8088 if "--ae" in argv else PORT
    args, panel, log, wait, jsx = [], False, False, 0.0, JSX
    it = iter(argv)
    for a in it:
        if a == "--panel":
            panel = True
        elif a == "--log":
            log = True
        elif a == "--wait":
            wait = float(next(it, 180))
        elif a == "--port":          # another panel: postRe is 8099 with its own jsx
            port = int(next(it))
        elif a == "--jsx":
            jsx = next(it)
        elif a == "--smoke":
            return smoke(next(it), port)
        elif not a.startswith("--"):
            args.append(a)

    if wait and not wait_for(port, wait):
        return sys.exit("devtools on :%d never came up in %ss" % (port, wait))
    if not args:
        print(__doc__)
        return 2

    logs: list = []
    code = sys.stdin.read() if args[0] == "-" else " ".join(args)
    out = call(code, panel=panel, port=port, logs=logs, jsx=jsx)
    print(out if out is not None else "(no value)")
    bad = isinstance(out, str) and out.startswith("Error")
    if logs and (log or bad):
        print("--- console ---\n" + "\n".join(logs), file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
