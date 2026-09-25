/* Browser stand-in for Premiere + CEP's Node, injected by preview.py. Never loaded in CEP.

   ?speed=N    replay the recorded ~40 min make N times faster (default 30)
   ?run=fail   the story pass dies with exit 1
   Cancel is the panel's real button; it kills the fake child like the real one.
   The replay's stage timings are the medians the panel's STAGES weights came from. */
(function () {
"use strict";

var q = new URLSearchParams(location.search);
var SPEED = +q.get("speed") || 30, FAIL = q.get("run") === "fail";
var REPO = "C:\\dev\\reFire", RUN = REPO + "\\run\\2872051525\\2872051525-wistful-otter";

window.__dirname = REPO + "\\reFire\\ppro";
window.process = {env: {}};

function Emitter() { this.h = {}; }
Emitter.prototype.on = function (ev, fn) { (this.h[ev] = this.h[ev] || []).push(fn); return this; };
Emitter.prototype.emit = function (ev, x) { (this.h[ev] || []).forEach(function (f) { f(x); }); };

function norm(p) {
    var out = [];
    String(p).replace(/\//g, "\\").split("\\").forEach(function (s) {
        if (s === "..") { out.pop(); } else if (s && s !== ".") { out.push(s); }
    });
    return out.join("\\");
}
var path = {
    normalize: norm,
    join: function () { return norm([].join.call(arguments, "\\")); },
    resolve: function () {
        return norm([].reduce.call(arguments, function (acc, p) {
            return /^([a-zA-Z]:|\\\\|\/)/.test(p) ? p : acc + "\\" + p;
        }, ""));
    }
};

// --- the replayed `refire make` -------------------------------------------
function pct(n, msg, t) { return "[" + ("  " + n).slice(-3) + "%] " + msg + "  (+" + t.toFixed(1) + "m)"; }

function makeScript(render) {
    var s = [], t = 0, i;
    function out(dt, text) { t += dt; s.push([t, "stdout", text]); }
    function err(text) { s.push([t, "stderr", text]); }
    function p(dt, n, msg) { t += dt; s.push([t, "stdout", pct(n, msg, t)]); }

    out(0, "[run] 2872051525-wistful-otter  (cache: run\\2872051525, artifacts: run\\2872051525\\2872051525-wistful-otter)");
    p(0, 0, "downloading VOD -- fetching video info");
    for (i = 0; i <= 100; i += 10) { p(0.35, Math.round(i / 10), "downloading VOD -- downloading " + i + "%"); }
    p(4, 10, "transcribing");
    for (i = 13; i <= 55; i += 3) { p(0.1, i, "transcribing"); }
    out(0, "[transcribe] 41822 words, ~139 min of speech, 164 chunks");
    p(8.5, 56, "fusing perception signals");
    p(0.1, 60, "writing story outline");
    out(0, "[director] stream map ~61k tokens, ~9 beats");
    p(0, 61, "scouting chapters");
    var ch = ["Morning commissions", "Wordle detour", "Boss attempt one", "Chat argues builds",
              "Artifact farming", "Rage at the RNG", "Lunch break", "Second boss attempt",
              "The clutch", "Victory lap", "Gacha pulls", "Viewer questions", "Signing off"];
    for (i = 0; i < ch.length; i++) {
        p(0.09, 61 + Math.floor(3 * (i + 1) / ch.length), "chapter " + (i + 1) + "/13 done 5s - " + ch[i]);
    }
    out(0, "[scout] 13 chapters");
    out(0, "[director] -> claude-opus-5 via claude CLI (subscription, new session, effort=xhigh): system 18,204 chars, user 244,118 chars");
    if (FAIL) {
        p(2, 100, "error: director pass failed: claude CLI exited 1 (usage limit reached)");
        err("director pass failed: claude CLI exited 1 (usage limit reached)");
        err("Refusing to fall back to flat selection -- it ignores every structural");
        s.exit = 1;
        return s;
    }
    out(8, "[director] <- stop_reason=end_turn, 9,812 chars");
    var beats = ["Cold open", "The commission grind", "Wordle spiral", "First boss attempt",
                 "Rage quit", "Coming back", "Second attempt", "The clutch", "Button"];
    function cast(first) {
        for (var b = 0; b < beats.length; b++) {
            p(0.08, first ? 66 + Math.round(24 * (b + 1) / beats.length) : 90,
              "casting beat " + (b + 1) + "/9: " + beats[b]);
        }
    }
    p(0, 66, "casting beats");
    cast(true);
    out(0.2, "[review] -> claude-opus-5 via claude CLI (subscription, resume, effort=xhigh): user 12,480 chars");
    out(6, "[review] round 1: revising -- the wordle spiral opens mid-sentence; pull its setup in");
    cast(false);
    out(0.2, "[review] -> claude-opus-5 via claude CLI (subscription, resume, effort=xhigh): user 11,902 chars");
    out(5.5, "[review] round 2: approved -- reads as one story now");
    p(0.2, 90, "selecting clips");
    p(0.1, 93, "placing overlays + music");
    p(0.1, 95, "building manifest");
    p(0.5, 96, "fixing captions");
    out(0, "[captions] -> claude-opus-5 via claude CLI (subscription, new session, effort=xhigh)");
    out(3, "[captions] <- stop_reason=end_turn, 3,112 chars");
    if (render) {
        p(0, 97, "rendering rough cut (ffmpeg)");
        out(6, "Rough cut: " + RUN + "\\rough.mp4");
    }
    p(0, 100, "done");
    out(0, "Manifest: " + RUN + "\\ae\\manifest.json");
    s.exit = 0;
    return s;
}

function spawn(py, args) {
    var cmd = args[2], proc = new Emitter(), timers = [];
    proc.stdout = new Emitter();
    proc.stderr = new Emitter();
    var script = cmd === "make" ? makeScript(args.indexOf("--render") >= 0)
               : cmd === "srt" ? [[0.005, "stdout", "Captions: " + RUN + "\\ae\\captions.srt"]]
               : [[0.02, "stdout", "[recap] 38 clip(s) on V1, 312 cues"],
                  [0.03, "stdout", "Captions: " + RUN + "\\ae\\captions.recut1.srt"]];
    var k = cmd === "make" ? SPEED : 1, end = script[script.length - 1][0];
    script.forEach(function (row) {
        timers.push(setTimeout(function () { proc[row[1]].emit("data", row[2] + "\n"); },
                               row[0] * 60000 / k));
    });
    timers.push(setTimeout(function () { proc.emit("close", script.exit || 0); },
                           end * 60000 / k + 50));
    proc.kill = function () {
        timers.forEach(clearTimeout);
        setTimeout(function () { proc.emit("close", null); }, 50);
    };
    return proc;
}

// --- Premiere: canned replies in reFirePpro.jsx's own words ----------------
window.__adobe_cep__ = {evalScript: function (src, cb) {
    var r = /reFirePpro\.probe\(/.test(src) ? "reFirePpro.jsx ok - Premiere 26.0 (preview), project 'preview'"
          : /reFirePpro\.build\(/.test(src) ? "Built 'reFire cut 3': 9 clip(s) in 41 cut(s), 1187.4s.  captions.srt is in the reFire bin -- drag it onto the sequence."
          : /reFirePpro\.timeline\(/.test(src) ? "Timeline: 38 clip(s) on V1 of 'reFire cut 3'."
          : /reFirePpro\.attach\(/.test(src) ? "Recaptioned: captions.recut1.srt is in the reFire bin. Delete the old caption track, then drag this one onto the sequence."
          : "Error: the preview has no canned reply for this call";
    setTimeout(function () { cb(r); }, 400);
}};
window.cep = {fs: {showOpenDialog: function () { return {data: [RUN + "\\ae\\manifest.json"]}; }}};

var mods = {
    fs: {existsSync: function () { return true; }, realpathSync: function (p) { return p; }},
    path: path,
    child_process: {spawn: spawn}
};
window.cep_node = {require: function (m) { return mods[m]; }};

// --- live reload ----------------------------------------------------------
var seen = null;
setInterval(function () {
    fetch("/__mtime").then(function (r) { return r.text(); }).then(function (t) {
        if (seen && t !== seen) { location.reload(); }
        seen = t;
    }).catch(function () {});
}, 1000);
}());
