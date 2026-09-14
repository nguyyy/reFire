"""Scout pass: map windowing, chapterize (mocked Ollama), digest assembly."""
import json

from refire import director, perception
from refire.director import Chapter, Notable, _parse_stamp, _windows

MAP = "\n".join(f"[{t}s] line at {t}." for t in range(0, 3600, 60))


def test_parse_stamp():
    assert _parse_stamp("[546s] hello") == 546.0
    assert _parse_stamp("no stamp") is None
    assert _parse_stamp("[12s] [scene: boss]") == 12.0


def test_windows_split_by_time():
    wins = _windows(MAP, window_s=1200.0)
    assert len(wins) == 3
    assert wins[0][:2] == (0.0, 1200.0)
    assert wins[1][:2] == (1200.0, 2400.0)
    assert "[0s]" in wins[0][2] and "[1140s]" in wins[0][2]
    assert "[1200s]" in wins[1][2]
    assert _windows("", 1200.0) == []


def _fake_client(calls, seen=None):
    """Scout calls go through `ollama.Client(timeout=...)`, not module-level `chat`."""
    class _Client:
        def __init__(self, *a, **k):
            if seen is not None:
                seen.update(k)

        def chat(self, model, format, messages, options):
            calls.append(messages[1]["content"])
            return {"message": {"content": json.dumps({
                "title": "Boss attempts", "summary": "He tries the boss.",
                "notable_moments": [{"t_s": 100.0, "why": "first wipe"},
                                    {"t_s": 99999.0, "why": "outside window"}]})}}
    return _Client


def test_chapterize_uses_ollama_and_caches(tmp_path, monkeypatch):
    calls = []
    import ollama
    monkeypatch.setattr(ollama, "Client", _fake_client(calls))
    cache = tmp_path / "chapters.json"
    chs = director.chapterize(MAP, window_s=1200.0, cache_path=cache)
    assert len(chs) == 3 and len(calls) == 3
    assert chs[0].title == "Boss attempts"
    assert chs[0].start_s == 0.0 and chs[0].end_s == 1200.0
    # hallucinated out-of-window stamp is clamped away
    assert [n.t_s for n in chs[0].notable_moments] == [100.0]
    # cache hit -> no new ollama calls
    again = director.chapterize(MAP, window_s=1200.0, cache_path=cache)
    assert len(calls) == 3 and again[0].title == "Boss attempts"


def test_chapterize_stub_on_failure(monkeypatch):
    import ollama

    class _Down:
        def __init__(self, *a, **k):
            pass

        def chat(self, **k):
            raise RuntimeError("down")

    monkeypatch.setattr(ollama, "Client", _Down)
    chs = director.chapterize(MAP, window_s=3600.0)
    assert len(chs) == 1
    assert chs[0].title.startswith("[0s] line at 0.")   # stub from first line
    assert chs[0].notable_moments == []


def test_the_scout_call_is_given_a_timeout(monkeypatch):
    """`ollama.chat` blocks forever, so the stub-chapter fallback above is dead code
    against a wedged runner unless the call is bounded."""
    import ollama
    seen = {}
    monkeypatch.setattr(ollama, "Client", _fake_client([], seen))
    director.chapterize(MAP, window_s=1200.0)
    assert seen.get("timeout") == director.SCOUT_TIMEOUT_S


def test_chapterize_reports_progress_before_and_after_each_window(monkeypatch):
    """The call is the multi-minute part: a report only on completion leaves the bar
    frozen for exactly as long as the user is waiting."""
    import ollama
    monkeypatch.setattr(ollama, "Client", _fake_client([]))
    seen = []
    director.chapterize(MAP, window_s=1200.0, progress=lambda f, m: seen.append((f, m)))

    assert len(seen) == 6                       # 3 windows, start + finish each
    assert seen[0][1].startswith("scouting chapter 1/3")
    assert "0.00h-0.33h" in seen[0][1]
    assert seen[1][1].startswith("chapter 1/3 done")
    assert "Boss attempts" in seen[1][1] and "left" in seen[1][1]
    assert [f for f, _ in seen] == sorted(f for f, _ in seen)     # never goes backwards
    assert seen[-1][0] == 1.0
    # cp1252 console: a non-ASCII glyph here raises UnicodeEncodeError mid-run
    assert all(m.isascii() for _, m in seen)


CHAPTERS = [
    Chapter(title="Setup", start_s=0, end_s=1800, summary="Calm start.",
            notable_moments=[Notable(t_s=600, why="first joke")]),
    Chapter(title="Chaos", start_s=1800, end_s=3540, summary="It goes wrong."),
]


def test_chapter_guide_lists_chapters_and_notables():
    g = perception.chapter_guide(CHAPTERS)
    assert "01. [0s-1800s] Setup -- Calm start." in g
    assert "* [600s] first joke" in g
    assert "02. [1800s-3540s] Chaos" in g


def test_excerpt_selects_stamped_range():
    ex = perception.excerpt(MAP, 100, 200)
    assert ex == "[120s] line at 120.\n[180s] line at 180."


def test_read_note_points_at_chapter_files():
    note = director._read_note()
    assert "map/chapter_NN.txt" in note
    assert f"{director.MAX_READS} Reads" in note


def test_realized_script_renders_beats_without_cut_times():
    log = {"central_idea": "x", "beats": [
        {"title": "A", "start": 100.0, "end": 130.0, "dur": 30.0, "text": "hi."},
        {"title": "B", "start": 400.0, "end": 407.0, "dur": 7.0, "text": "yo."}]}
    s = director._realized_script(log)
    assert "A" in s and "hi." in s and "B" in s
    assert "in the cut" not in s


def test_digest_contains_guide_and_bounded_excerpts():
    d = perception.digest(CHAPTERS, MAP, windows=[(3000, 3120)], pad_s=30.0)
    assert d.startswith("CHAPTER GUIDE")
    assert "--- excerpt [570s-630s] ---" in d      # notable +-30
    assert "--- excerpt [2970s-3150s] ---" in d    # top window +-30
    assert "[600s] line at 600." in d
    # a tiny budget keeps the guide but drops excerpts
    tiny = perception.digest(CHAPTERS, MAP, windows=[(3000, 3120)],
                             budget_chars=len(perception.chapter_guide(CHAPTERS)) + 120)
    assert "CHAPTER GUIDE" in tiny and "line at 3000." not in tiny
