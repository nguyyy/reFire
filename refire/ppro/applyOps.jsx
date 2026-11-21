// Apply an ops_<feature>.json written by python (refire.punch.ops, refire.caption.ops).
// Thin and dumb on purpose: no decisions, every time/value comes from the file. Idempotent
// because python always starts the list with a clear of the items reFire tagged. ES3.
// Spike findings baked in (docs/spikes.md): QE only on the active sequence, QE items matched
// by start ticks (its lists include gaps), effects verified by matchName after adding,
// project items matched by trimmed name.

var reFireOps = (function () {

    function q(s) {
        return '"' + String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n") + '"';
    }
    function T(ticks) { var t = new Time(); t.ticks = String(ticks); return t; }
    function S(sec) { var t = new Time(); t.seconds = sec; return t; }
    function seq() { return app.project.activeSequence; }

    // per apply() call: the sequence that was active (= main), nests whose V5 clip was just placed
    var MAIN = null, FRESH = {};
    function byName(name) {
        for (var i = 0; i < app.project.sequences.numSequences; i++) {
            if (app.project.sequences[i].name === name) { return app.project.sequences[i]; }
        }
        return null;
    }
    function activate(s) {
        if (!seq() || seq().sequenceID !== s.sequenceID) { app.project.activeSequence = s; }
        return s;
    }
    // ops that name a sequence run there (QE needs it active); the rest run on main
    function onSeq(op) {
        var s = op.seq ? byName(op.seq) : MAIN;
        if (!s) { throw new Error("no sequence named " + op.seq); }
        return activate(s);
    }
    // importMGT fails ~15% of calls in bursts, placing nothing (S6): verify + retry
    function importAt(s, path, ticks, v) {
        var tr = s.videoTracks[v], k, ti;
        for (k = 0; k < 6; k++) {
            var before = tr.clips.numItems;
            ti = s.importMGT(path, String(ticks), v, 0);
            if (tr.clips.numItems === before + 1 && ti && Number(ti.start.ticks) === Number(ticks)) { return ti; }
            $.sleep(100);
        }
        throw new Error("importMGT kept failing at " + ticks + " on V" + (v + 1));
    }
    function clearClips(tr) {
        for (var c = tr.clips.numItems - 1; c >= 0; c--) { tr.clips[c].remove(false, false); }
    }
    function refersTo(clip, item) {
        return clip.projectItem && item && clip.projectItem.nodeId === item.nodeId;
    }
    function trimmed(s) { return String(s).replace(/^\s+|\s+$/g, ""); }

    function readJson(path) {
        var f = new File(path);
        f.encoding = "UTF-8";
        if (!f.open("r")) { throw new Error("cannot read " + path); }
        var s = f.read();
        f.close();
        return eval("(" + s + ")");          // es3 has no JSON; the file is our own output
    }
    function findItem(name, node) {
        node = node || app.project.rootItem;
        for (var i = 0; i < node.children.numItems; i++) {
            var it = node.children[i];
            if (trimmed(it.name) === name) { return it; }
            if (it.type === ProjectItemType.BIN) {
                var r = findItem(name, it);
                if (r) { return r; }
            }
        }
        return null;
    }
    function itemAt(track, ticks) {
        for (var i = 0; i < track.clips.numItems; i++) {
            if (Number(track.clips[i].start.ticks) === Number(ticks)) { return track.clips[i]; }
        }
        return null;
    }
    function qeItemAt(v, ticks) {
        app.enableQE();
        var tr = qe.project.getActiveSequence().getVideoTrackAt(v), i, it;
        for (i = 0; i < tr.numItems; i++) {
            it = tr.getItemAt(i);
            if (it && it.type !== "Empty" && Number(it.start.ticks) === Number(ticks)) { return it; }
        }
        return null;
    }
    function component(ti, match) {
        for (var i = ti.components.numItems - 1; i >= 0; i--) {   // newest instance last
            if (ti.components[i].matchName === match) { return ti.components[i]; }
        }
        return null;
    }
    function count(ti, match) {
        var n = 0;
        for (var i = 0; i < ti.components.numItems; i++) { if (ti.components[i].matchName === match) { n++; } }
        return n;
    }
    function track(v) {
        var s = activate(MAIN);
        if (s.videoTracks.numTracks <= v) {
            app.enableQE();
            qe.project.getActiveSequence().addTracks(v + 1 - s.videoTracks.numTracks, s.videoTracks.numTracks, 0);
        }
        return s.videoTracks[v];
    }

    function addEffect(ti, qi, name, match) {
        var n0 = count(ti, match);
        qi.addVideoEffect(qe.project.getVideoEffectByName(name));
        // an unknown name adds nothing and says nothing (S7), so check what landed
        if (count(ti, match) <= n0) { throw new Error("'" + name + "' did not add " + match + " (renamed in this Premiere?)"); }
        return component(ti, match);
    }
    function setParams(c, params) {
        for (var i = 0; i < params.length; i++) {
            var p = c.properties[params[i][0]], v = params[i][1];
            if (v && v.color) { p.setColorValue(v.color[0], v.color[1], v.color[2], v.color[3], true); }
            else { p.setValue(v, true); }
        }
    }
    // keys: [[param index, [[seconds from clip start, value], ...]], ...]
    function setKeys(c, ti, keys, interp) {
        var base = ti.inPoint.seconds, i, k;
        for (i = 0; i < keys.length; i++) {
            var p = c.properties[keys[i][0]], list = keys[i][1];
            p.setTimeVarying(true);
            for (k = 0; k < list.length; k++) {
                var t = S(base + list[k][0]);
                p.addKey(t);
                p.setValueAtKey(t, list[k][1], false);
                p.setInterpolationTypeAtKey(t, interp, false);
            }
        }
    }
    // main's end ignoring our own nest clip, so the nest follows a re-cut that shortened main
    function mainEnd(m, nestItem) {
        var end = 0, g, t, c, groups = [m.videoTracks, m.audioTracks];
        for (g = 0; g < 2; g++) {
            for (t = 0; t < groups[g].numTracks; t++) {
                for (c = 0; c < groups[g][t].clips.numItems; c++) {
                    var clip = groups[g][t].clips[c];
                    if (!refersTo(clip, nestItem)) { end = Math.max(end, Number(clip.end.ticks)); }
                }
            }
        }
        return end;
    }

    var OPS = {
        clear_track: function (op) {
            var tr = track(op.v), n = 0;
            for (var c = tr.clips.numItems - 1; c >= 0; c--) {
                var pi = tr.clips[c].projectItem;
                if (pi && trimmed(pi.name) === op.item) { tr.clips[c].remove(false, false); n++; }
            }
            return n;
        },
        insert_adjustment: function (op) {
            var item = findItem(op.item);
            if (!item) {
                throw new Error("no project item '" + op.item + "': File > New > Adjustment Layer, rename it " + op.item);
            }
            var tr = track(op.v);
            tr.overwriteClip(item, T(op.start));
            var ti = itemAt(tr, op.start);
            if (!ti) { throw new Error("adjustment layer did not land at " + op.start); }
            ti.end = T(op.end);
            if (op.name) { ti.name = op.name; }
            return 1;
        },
        add_effect: function (op) {
            var s = onSeq(op), qi = qeItemAt(op.v, op.start), ti = itemAt(s.videoTracks[op.v], op.start);
            if (!qi || !ti) { throw new Error("no clip on V" + (op.v + 1) + " at " + op.start + " in " + s.name); }
            var c = addEffect(ti, qi, op.effect, op.match);
            setParams(c, op.params || []);
            if (op.keys) { setKeys(c, ti, op.keys, op.interp); }
            return 1;
        },
        // the caption nest: create it once (clone main, cleared), keep an invisible anchor at
        // main's end - 1 frame (an empty nest can't be placed, S5), and exactly one clip of it
        // on main V(op.v) spanning main. op.fresh replaces that clip (restyle)
        ensure_nest: function (op) {
            var m = activate(MAIN), i, g, t;
            if (m.name === op.nest) { throw new Error("open your main sequence, not " + op.nest); }
            var nest = byName(op.nest);
            if (!nest) {
                var before = {};
                for (i = 0; i < app.project.sequences.numSequences; i++) { before[app.project.sequences[i].sequenceID] = 1; }
                m.clone();                                    // makes the clone active (S5)
                for (i = 0; i < app.project.sequences.numSequences; i++) {
                    if (!before[app.project.sequences[i].sequenceID]) { nest = app.project.sequences[i]; }
                }
                if (!nest) { throw new Error("cloning " + m.name + " made no sequence"); }
                nest.name = op.nest;
                var groups = [nest.videoTracks, nest.audioTracks];
                for (g = 0; g < 2; g++) { for (t = 0; t < groups[g].numTracks; t++) { clearClips(groups[g][t]); } }
                activate(m);
            }
            if (nest.videoTracks.numTracks < op.tracks) {
                activate(nest);
                qe.project.getActiveSequence().addTracks(op.tracks - nest.videoTracks.numTracks, nest.videoTracks.numTracks, 0);
                activate(m);
            }
            var end = mainEnd(m, nest.projectItem), top = op.tracks - 1;
            clearClips(nest.videoTracks[top]);
            var anc = importAt(nest, op.anchor, end - Number(m.timebase), top);
            anc.end = T(end);
            anc.name = "refire:caption:anchor";
            var tr = track(op.v), found = [];
            for (i = tr.clips.numItems - 1; i >= 0; i--) {
                if (refersTo(tr.clips[i], nest.projectItem)) { found.push(tr.clips[i]); }
            }
            for (i = op.fresh ? 0 : 1; i < found.length; i++) { found[i].remove(false, false); }
            var clip = op.fresh ? null : found[0];
            if (!clip) {
                tr.overwriteClip(nest.projectItem, T(0));
                clip = itemAt(tr, 0);
                if (!clip) { throw new Error(op.nest + " did not land on V" + (op.v + 1)); }
                FRESH[op.nest] = true;
            }
            clip.end = T(end);
            return 1;
        },
        // ccStyle on the nest clip, only when that clip was just placed (new nest or restyle)
        style_nest: function (op) {
            if (!FRESH[op.nest]) { return 0; }
            activate(MAIN);
            var tr = MAIN.videoTracks[op.v], ti = itemAt(tr, 0), qi = qeItemAt(op.v, 0), i;
            if (!ti || !qi) { throw new Error("no " + op.nest + " clip at 0 on V" + (op.v + 1)); }
            for (i = 0; i < op.effects.length; i++) {
                setParams(addEffect(ti, qi, op.effects[i].effect, op.effects[i].match), op.effects[i].params);
            }
            return 1;
        },
        clear_nest: function (op) {
            var nest = byName(op.nest);
            for (var v = 0; v < op.tracks && v < nest.videoTracks.numTracks; v++) { clearClips(nest.videoTracks[v]); }
            return 1;
        },
        insert_caption: function (op) {
            var s = byName(op.seq);                          // importMGT works on a non-active sequence (S5)
            var ti = importAt(s, op.mogrt, op.start, op.v);
            ti.end = T(op.end);
            if (op.name) { ti.name = op.name; }
            return 1;
        }
    };

    // apply ops[from .. from+count) of the file; -> json summary. chunked so long lists keep
    // the panel responsive (S6) and a failure names the op
    function apply(path, from, count) {
        if (!seq()) { return '{"error": "no active sequence"}'; }
        MAIN = seq();
        FRESH = {};
        var all = readJson(path).ops, end = Math.min(all.length, from + count), done = 0, i, err = null;
        for (i = from; i < end; i++) {
            try {
                OPS[all[i].op](all[i]);
                done++;
            } catch (e) {
                err = "op " + i + " (" + all[i].op + (all[i].id ? " " + all[i].id : "") + "): " + (e.message || e);
                break;
            }
        }
        try { activate(MAIN); } catch (e2) {}               // nest work leaves the nest active
        if (err) { return '{"done": ' + done + ', "total": ' + all.length + ', "error": ' + q(err) + '}'; }
        return '{"done": ' + done + ', "next": ' + end + ', "total": ' + all.length + '}';
    }

    // never throw out of here: CEP turns that into a blank "EvalScript error."
    return { apply: function (path, from, count) {
        try { return apply(path, from, count); } catch (e) { return '{"error": ' + q(e.message || e) + '}'; }
    } };
}());
