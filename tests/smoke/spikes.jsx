// Phase 0 API spikes (docs/specs/00-shared.md S1-S7). Throwaway probes, ES3, driven by
// tests/smoke/spikes.py through remote.py. Every function returns a JSON string so the
// python side can assert on it. Not production code: findings go to docs/spikes.md and
// the real applier is written afterwards in refire/ppro/.

var spike = (function () {

    // --- tiny ES3 json (extendscript has no JSON object) ------------------------
    function q(s) {
        return '"' + String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"')
            .replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t") + '"';
    }
    function J(v) {
        var i, out, k;
        if (v === null || v === undefined) { return "null"; }
        if (typeof v === "number") { return isFinite(v) ? String(v) : "null"; }
        if (typeof v === "boolean") { return v ? "true" : "false"; }
        if (typeof v === "string") { return q(v); }
        if (v instanceof Array) {
            out = [];
            for (i = 0; i < v.length; i++) { out.push(J(v[i])); }
            return "[" + out.join(",") + "]";
        }
        out = [];
        for (k in v) { if (v.hasOwnProperty(k)) { out.push(q(k) + ":" + J(v[k])); } }
        return "{" + out.join(",") + "}";
    }
    function safe(fn) {
        try { return fn(); } catch (e) { return "ERR " + (e.message || e) + (e.line ? " @" + e.line : ""); }
    }

    // --- helpers ----------------------------------------------------------------
    function T(ticks) { var t = new Time(); t.ticks = String(ticks); return t; }
    function S(sec) { var t = new Time(); t.seconds = sec; return t; }
    function seq() { return app.project.activeSequence; }
    function tpf() { return Number(seq().timebase); }          // ticks per frame
    function snap(sec) { var f = tpf(); return Math.round(sec * 254016000000 / f) * f; }

    function findItem(name, node) {
        node = node || app.project.rootItem;
        for (var i = 0; i < node.children.numItems; i++) {
            var it = node.children[i];
            if (it.name === name) { return it; }
            if (it.type === ProjectItemType.BIN) {
                var r = findItem(name, it);
                if (r) { return r; }
            }
        }
        return null;
    }
    function itemAt(track, startTicks) {
        for (var i = 0; i < track.clips.numItems; i++) {
            if (Number(track.clips[i].start.ticks) === Number(startTicks)) { return track.clips[i]; }
        }
        return null;
    }
    function qeSeq() { app.enableQE(); return qe.project.getActiveSequence(); }
    // QE track lists include empty gaps, so match by start ticks, never by index
    function qeItemAt(vIdx, startTicks) {
        var tr = qeSeq().getVideoTrackAt(vIdx), i, it;
        for (i = 0; i < tr.numItems; i++) {
            it = tr.getItemAt(i);
            if (it && it.type !== "Empty" && Number(it.start.ticks) === Number(startTicks)) { return it; }
        }
        return null;
    }
    function effectNames() {
        app.enableQE();
        var l = qe.project.getVideoEffectList(), out = [], i;
        if (typeof l === "string") { return l.split(/[\n;,]+/); }
        for (i = 0; i < l.length; i++) { out.push(String(l[i])); }
        return out;
    }
    function addEffect(vIdx, startTicks, name) {
        var it = qeItemAt(vIdx, startTicks);
        if (!it) { throw new Error("no QE item on V" + (vIdx + 1) + " at " + startTicks); }
        var fx = qe.project.getVideoEffectByName(name);
        if (!fx) { throw new Error("no effect named " + name); }
        it.addVideoEffect(fx);
    }
    function component(ti, matchName) {
        for (var i = ti.components.numItems - 1; i >= 0; i--) {   // newest instance last
            if (ti.components[i].matchName === matchName) { return ti.components[i]; }
        }
        return null;
    }
    function dumpProps(c) {
        var out = [], j, p, v;
        for (j = 0; j < c.properties.numItems; j++) {
            p = c.properties[j];
            v = safe(function () { return p.getValue(); });
            out.push({ j: j, name: p.displayName, value: (v && v.length !== undefined && typeof v !== "string") ? J(v) : v });
        }
        return out;
    }
    function frame(sec, path) {
        var s = seq();
        s.setPlayerPosition(String(snap(sec)));
        var qs = qeSeq();
        qs.exportFramePNG(qs.CTI.timecode, path);
        return path;
    }
    function ensureVideoTracks(n) {
        var s = seq(), have = s.videoTracks.numTracks;
        if (have < n) { qeSeq().addTracks(n - have, have, 0); }
        return s.videoTracks.numTracks;
    }
    function clearTrack(track) {
        for (var c = track.clips.numItems - 1; c >= 0; c--) { track.clips[c].remove(false, false); }
    }

    // --- setup: a scratch sequence from one proxy, >= 5 video tracks -------------
    function setup(proxyPath) {
        app.project.importFiles([proxyPath], true, app.project.rootItem, false);
        var f = new File(proxyPath), item = null, root = app.project.rootItem, i;
        for (i = 0; i < root.children.numItems; i++) {
            var mp = safe(function () { return root.children[i].getMediaPath(); });
            if (String(mp).replace(/\\/g, "/").toLowerCase() === f.fsName.replace(/\\/g, "/").toLowerCase()) {
                item = root.children[i];
            }
        }
        if (!item) { throw new Error("import failed"); }
        app.project.createNewSequenceFromClips("refire_spike", [item], root);
        var tracks = ensureVideoTracks(5);
        var s = seq();
        return J({ seq: s.name, fps_ticks: s.timebase, end: s.end, v1: s.videoTracks[0].clips.numItems,
                   vtracks: tracks, ppro: app.version });
    }

    // --- S1: adjustment layer on V2 at exact time + length ------------------------
    function s1(name, at, lens) {
        var item = findItem(name), res = { item: !!item, placed: [] }, i;
        if (!item) {
            res.qe_new_adjustment = safe(function () { app.enableQE(); return typeof qe.project.newAdjustmentLayer; });
            return J(res);
        }
        var v2 = seq().videoTracks[1];
        clearTrack(v2);
        var t = at;
        for (i = 0; i < lens.length; i++) {
            var a = snap(t), b = snap(t + lens[i]);
            safe(function () { item.setOutPoint(lens[i] + 1, 1); });
            v2.overwriteClip(item, T(a));
            var ti = itemAt(v2, a);
            var setEnd = safe(function () { ti.end = T(b); return "ok"; });
            var nm = safe(function () { ti.name = "refire:punch:s1_" + i; return ti.name; });
            res.placed.push({ want: [a, b], got: ti ? [Number(ti.start.ticks), Number(ti.end.ticks)] : null,
                              set_end: setEnd, name: nm });
            t += lens[i] + 2;
        }
        return J(res);
    }

    // --- S2 + S7: QE Transform on the first V2 item, pinned zoom -----------------
    function s2(startSec, scale, pin, outDir) {
        var a = snap(startSec), v2 = seq().videoTracks[1], ti = itemAt(v2, a);
        if (!ti) { throw new Error("no V2 item at " + startSec + "s (run S1 first)"); }
        addEffect(1, a, "Transform");
        var c = component(ti, "AE.ADBE Geometry2");
        var before = dumpProps(c);
        // by preset Params Index: 0 anchor, 1 position, 2 uniform(11), 3 h, 4 w, ... 9 use-comp-shutter(9)
        var res = { names: effectNames().length, before: before };
        res.set = safe(function () {
            c.properties[0].setValue(pin, true);
            c.properties[1].setValue(pin, true);
            c.properties[2].setValue(true, true);
            c.properties[3].setValue(scale, true);
            return "ok";
        });
        res.after = dumpProps(c);
        var mid = (Number(ti.start.ticks) + Number(ti.end.ticks)) / 2 / 254016000000;
        res.frame_zoom = frame(mid, outDir + "/s2_zoom.png");
        res.frame_plain = frame(mid + (ti.end.seconds - ti.start.seconds), outDir + "/s2_plain.png");
        return J(res);
    }

    // toggle one property and read it back (S7: which index is which checkbox)
    function toggle(vIdx, startSec, matchName, j, val) {
        var ti = itemAt(seq().videoTracks[vIdx], snap(startSec)), c = component(ti, matchName);
        c.properties[j].setValue(val, true);
        return J({ j: j, name: c.properties[j].displayName, value: c.properties[j].getValue() });
    }

    // --- S3: importMGT + TextLayer ----------------------------------------------
    function s3(mogrt, atSec, vIdx, lenSec, textJson, outDir) {
        var s = seq(), a = snap(atSec);
        var ti = s.importMGT(mogrt, String(a), vIdx, 0);
        if (!ti) { throw new Error("importMGT returned nothing"); }
        var res = { start: Number(ti.start.ticks), end0: Number(ti.end.ticks) };
        res.set_end = safe(function () { ti.end = T(snap(atSec + lenSec)); return Number(ti.end.ticks); });
        var mgt = ti.getMGTComponent(), p = mgt.properties.getParamForDisplayName("TextLayer");
        res.params = [];
        for (var j = 0; j < mgt.properties.numItems; j++) { res.params.push(mgt.properties[j].displayName); }
        var v = String(p.getValue());
        res.value_before = v;
        var nv = v.replace(/"textEditValue":"(?:[^"\\]|\\.)*"/, '"textEditValue":' + textJson);
        p.setValue(nv, true);
        res.value_after = String(p.getValue());
        res.frame = frame(atSec + lenSec / 2, outDir + "/s3_text.png");
        return J(res);
    }

    // --- S4: Transform with bezier keys + motion blur on a MOGRT clip ------------
    // keys: {opacity: [[t, v], ...], position: [[t, [x, y]], ...]} in clip-relative seconds
    function s4(vIdx, atSec, keys, interp, shutterOff, outDir, frames) {
        var a = snap(atSec), ti = itemAt(seq().videoTracks[vIdx], a);
        if (!ti) { throw new Error("no clip at " + atSec); }
        addEffect(vIdx, a, "Transform");
        var c = component(ti, "AE.ADBE Geometry2"), res = { keys: [] };
        var base = ti.inPoint.seconds;            // key times are in the clip's own media time
        function keyed(j, list) {
            var p = c.properties[j], k;
            p.setTimeVarying(true);
            for (k = 0; k < list.length; k++) {
                var t = S(base + list[k][0]);
                p.addKey(t);
                p.setValueAtKey(t, list[k][1], true);
                res.keys.push(safe(function () { p.setInterpolationTypeAtKey(t, interp, true); return "ok"; }));
            }
        }
        c.properties[0].setValue([0.5, 0.5], true);
        keyed(1, keys.position);
        keyed(8, keys.opacity);
        if (shutterOff) { c.properties[9].setValue(false, true); }
        c.properties[10].setValue(360, true);
        res.props = dumpProps(c);
        res.frames = [];
        for (var i = 0; i < frames.length; i++) {
            res.frames.push(frame(atSec + frames[i], outDir + "/s4_" + vIdx + "_" + i + ".png"));
        }
        return J(res);
    }

    // --- S5: the caption nest on V5 + ccStyle effects ------------------------------
    function s5(nestName) {
        var main = seq(), res = { main: main.name };
        var nest = null, i;
        for (i = 0; i < app.project.sequences.numSequences; i++) {
            if (app.project.sequences[i].name === nestName) { nest = app.project.sequences[i]; }
        }
        if (!nest) {
            res.clone = safe(function () { return main.clone(); });
            for (i = 0; i < app.project.sequences.numSequences; i++) {
                var sq = app.project.sequences[i];
                if (sq.name !== main.name && sq.name.indexOf(main.name) === 0) { nest = sq; }
            }
            if (!nest) { throw new Error("clone produced no sequence"); }
            nest.name = nestName;
            var g = [nest.videoTracks, nest.audioTracks], t;
            for (i = 0; i < 2; i++) { for (t = 0; t < g[i].numTracks; t++) { clearTrack(g[i][t]); } }
        }
        res.nest = nest.name;
        res.vtracks = ensureVideoTracks(5);
        var v5 = main.videoTracks[4];
        clearTrack(v5);
        v5.overwriteClip(nest.projectItem, T(0));
        var ti = v5.clips[0];
        res.set_end = safe(function () { ti.end = T(main.end); return Number(ti.end.ticks); });
        res.main_end = Number(main.end);
        var names = effectNames(), want = [/camera shake/i, /drop shadow/i, /edge glow/i], hit = [], k;
        for (k = 0; k < want.length; k++) {
            for (i = 0; i < names.length; i++) { if (want[k].test(names[i])) { hit.push(names[i]); break; } }
        }
        res.effect_names = hit;
        for (k = 0; k < hit.length; k++) {
            res["add_" + k] = safe(function () { addEffect(4, Number(ti.start.ticks), hit[k]); return "ok"; });
        }
        res.components = [];
        for (i = 0; i < ti.components.numItems; i++) {
            res.components.push({ match: ti.components[i].matchName, name: ti.components[i].displayName,
                                  props: dumpProps(ti.components[i]) });
        }
        return J(res);
    }

    function setProp(vIdx, startSec, matchName, j, value) {
        var ti = itemAt(seq().videoTracks[vIdx], snap(startSec)), c = component(ti, matchName);
        c.properties[j].setValue(value, true);
        return J({ j: j, name: c.properties[j].displayName, value: c.properties[j].getValue() });
    }
    function setColor(vIdx, startSec, matchName, j, a, r, g, b) {
        var ti = itemAt(seq().videoTracks[vIdx], snap(startSec)), c = component(ti, matchName);
        c.properties[j].setColorValue(a, r, g, b, true);
        return J({ j: j, name: c.properties[j].displayName,
                   value: safe(function () { return J(c.properties[j].getColorValue()); }) });
    }

    // --- S6: chunked bulk insert, timed ---------------------------------------------
    function s6(mogrt, blocks, keys) {
        var s = seq(), t0 = new Date().getTime(), n = 0, i;
        for (i = 0; i < blocks.length; i++) {
            var b = blocks[i], a = snap(b[0]);
            var ti = s.importMGT(mogrt, String(a), b[2], 0);
            ti.end = T(snap(b[1]));
            var p = ti.getMGTComponent().properties.getParamForDisplayName("TextLayer");
            var v = String(p.getValue());
            p.setValue(v.replace(/"textEditValue":"(?:[^"\\]|\\.)*"/, '"textEditValue":"blk ' + i + '"'), true);
            addEffect(b[2], a, "Transform");
            var c = component(ti, "AE.ADBE Geometry2"), base = ti.inPoint.seconds, k, pr;
            var groups = [[1, keys.position], [8, keys.opacity]];
            for (var g = 0; g < groups.length; g++) {
                pr = c.properties[groups[g][0]];
                pr.setTimeVarying(true);
                for (k = 0; k < groups[g][1].length; k++) {
                    var kt = S(base + groups[g][1][k][0]);
                    pr.addKey(kt);
                    pr.setValueAtKey(kt, groups[g][1][k][1], false);
                    pr.setInterpolationTypeAtKey(kt, 5, false);
                }
            }
            c.properties[9].setValue(false, false);
            n++;
        }
        return J({ n: n, ms: new Date().getTime() - t0 });
    }

    return {
        setup: function (p) { return safe(function () { return setup(p); }); },
        s1: function (n, a, l) { return safe(function () { return s1(n, a, l); }); },
        s2: function (a, s, p, o) { return safe(function () { return s2(a, s, p, o); }); },
        toggle: function (v, a, m, j, x) { return safe(function () { return toggle(v, a, m, j, x); }); },
        s3: function (m, a, v, l, t, o) { return safe(function () { return s3(m, a, v, l, t, o); }); },
        s4: function (v, a, k, i, sh, o, f) { return safe(function () { return s4(v, a, k, i, sh, o, f); }); },
        s5: function (n) { return safe(function () { return s5(n); }); },
        setProp: function (v, a, m, j, x) { return safe(function () { return setProp(v, a, m, j, x); }); },
        setColor: function (v, a, m, j, A, r, g, b) { return safe(function () { return setColor(v, a, m, j, A, r, g, b); }); },
        s6: function (m, b, k) { return safe(function () { return s6(m, b, k); }); },
        effects: function () { return safe(function () { return J(effectNames()); }); }
    };
}());
