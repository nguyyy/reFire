"""Thumbnail determinism golden (verify skill step 4): render tests/golden/thumb/spec.json twice
and compare against the committed hash. A changed hash must be intentional: rerun with
--update and say why in the commit. Needs Playwright's Chromium + the Conneqt font.

    uv run python tests/golden/thumb/check.py [--update]
"""
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from refire.thumb import render  # noqa: E402

HERE = Path(__file__).parent


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "c"
        shutil.copytree(HERE, d, ignore=shutil.ignore_patterns("check.py", "*.sha256", "thumb.png", "index.html"))
        a = render.render(d / "spec.json").read_bytes()
        b = render.render(d / "spec.json").read_bytes()
        shutil.copy(d / "thumb.png", HERE / "thumb.png")          # for eyeballing; gitignored
    got = hashlib.sha256(a).hexdigest()
    if a != b:
        print("FAIL: two renders of the same spec differ")
        return 1
    want_path = HERE / "thumb.sha256"
    if "--update" in sys.argv or not want_path.exists():
        want_path.write_text(got + "\n")
        print("golden updated:", got)
        return 0
    want = want_path.read_text().strip()
    print(("PASS" if got == want else "FAIL") + f": {got} (golden {want})")
    return 0 if got == want else 1


if __name__ == "__main__":
    raise SystemExit(main())
