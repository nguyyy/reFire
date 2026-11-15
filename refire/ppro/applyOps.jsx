// Apply an ops_<feature>.json written by python (refire.punch.ops, later captions).
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
    function seq() { return app.project.activeSequence; }
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
        var s = seq();
        if (s.videoTracks.numTracks <= v) {
            app.enableQE();
            qe.project.getActiveSequence().addTracks(v + 1 - s.videoTracks.numTracks, s.videoTracks.numTracks, 0);
        }
        return s.videoTracks[v];
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
            var qi = qeItemAt(op.v, op.start), ti = itemAt(seq().videoTracks[op.v], op.start);
            if (!qi || !ti) { throw new Error("no clip on V" + (op.v + 1) + " at " + op.start); }
            var n0 = count(ti, op.match), i;
            qi.addVideoEffect(qe.project.getVideoEffectByName(op.effect));
            // an unknown name adds nothing and says nothing (S7), so check what landed
            if (count(ti, op.match) <= n0) {
                throw new Error("'" + op.effect + "' did not add " + op.match + " (renamed in this Premiere?)");
            }
            var c = component(ti, op.match);
            for (i = 0; i < op.params.length; i++) {
                var p = c.properties[op.params[i][0]], v = op.params[i][1];
                if (v && v.color) { p.setColorValue(v.color[0], v.color[1], v.color[2], v.color[3], true); }
                else { p.setValue(v, true); }
            }
            return 1;
        }
    };

    // apply ops[from .. from+count) of the file; -> json summary. chunked so long lists keep
    // the panel responsive (S6) and a failure names the op
    function apply(path, from, count) {
        if (!seq()) { return '{"error": "no active sequence"}'; }
        var all = readJson(path).ops, end = Math.min(all.length, from + count), done = 0, i;
        for (i = from; i < end; i++) {
            try {
                OPS[all[i].op](all[i]);
                done++;
            } catch (e) {
                return '{"done": ' + done + ', "total": ' + all.length + ', "error": ' +
                       q("op " + i + " (" + all[i].op + (all[i].id ? " " + all[i].id : "") + "): " + (e.message || e)) + '}';
            }
        }
        return '{"done": ' + done + ', "next": ' + end + ', "total": ' + all.length + '}';
    }

    return { apply: apply };
}());
