// reFire — After Effects caption panel (ExtendScript, ES3).
//
// Launch as a dockable panel (drop in AE's "ScriptUI Panels" folder, then
// Window > reFire) or as a floating palette (File > Scripts > Run Script File...).
//
//   Choose manifest...  pick run/ae/manifest.json
//   Topic / Count       optional: stream subject (groups+orders clips into titled
//                        sections via the local LLM) and how many clips to keep.
//   Fine-tune           Min score drops weak/boring clips (higher = less footage);
//                        Zoom amount scales how readily motion triggers a punch
//                        (1 = normal, >1 = more/earlier zooms, <1 = fewer).
//   Generate            re-run `python -m refire ae` with the topic/count -> writes
//                        a fresh manifest (motion zoom episodes + sections), then
//                        reloads it. Needs python on PATH + a manifest already
//                        chosen (it supplies the video + run dir). Then click Build.
//   Build               import footage + build one comp per clip (footage trim,
//                        bottom-left zoom keyframes, captions) + a Master comp.
//   Update              re-apply captions + footage zoom on the already-built
//                        comps from the current templates and manifest, without
//                        re-importing footage. Use after tweaking either template
//                        or re-exporting the manifest; a new source needs Build.
//
// STYLE + ANIMATION: every caption is a clone of the text layer named
// "Caption Style" in the "Caption Template" comp. Restyle AND animate that ONE
// layer (font/colour/stroke + text animators / opacity keyframes), then click
// Update -- the look and the motion propagate to every caption. The template
// animation is stretched to fit each caption (first keyframe -> first word, last
// keyframe -> last word), plus a short fade-out at the last word.
//
// ZOOM: the punch SHAPE comes from the "Zoom Style" null's Scale in the "Zoom
// Template" comp (default 100 -> 190 -> 200 -> 100); WHERE it fires comes from each
// clip's motion episodes (manifest "zoom_episodes"). The template is stamped once
// per episode -- so a clip zooms only on interesting motion and can punch several
// times, holding at the 100% fit between episodes. Within an episode the punch-in
// (first segment) and punch-out (last) keep their authored durations; the middle
// creep stretches to fill. Edit that ONE null's Scale and Update to re-punch all.
//
// SECTIONS: with a manifest topic, clips are grouped into ordered sections. Each
// titled section drops a title card before its clips, and the Master crossfades
// neighbours. The card is a clone of the WHOLE "Section Card" comp -- every layer
// in it (the styled default is a dark backdrop + accent bar + the "Section Style"
// title text) is copied onto each card, with the title set on "Section Style".
// Restyle or add layers there, then Build or Update and they propagate to all cards.

(function (thisObj) {
    var manifestPath = null;   // remembered for the panel's lifetime
    var CARD_S = 1.5;          // section title-card duration (seconds)
    var XFADE_S = 0.25;        // crossfade between master layers (seconds)

    // --- manifest ----------------------------------------------------------

    function readManifest() {
        if (!manifestPath) { return null; }
        var f = new File(manifestPath);
        if (!f.exists) { return null; }
        f.open("r");
        var text = f.read();
        f.close();
        // ponytail: JSON is valid JS object-literal; eval avoids bundling json2.js.
        return eval("(" + text + ")");
    }

    // --- project helpers ---------------------------------------------------

    function findComp(name) {
        for (var i = 1; i <= app.project.numItems; i++) {
            var it = app.project.item(i);
            if (it instanceof CompItem && it.name === name) { return it; }
        }
        return null;
    }

    function findFootage(path) {
        // dedupe: reuse an already-imported source so repeat Builds don't clutter.
        var want = new File(path).fsName;
        for (var i = 1; i <= app.project.numItems; i++) {
            var it = app.project.item(i);
            if (it instanceof FootageItem && it.mainSource &&
                it.mainSource.file && it.mainSource.file.fsName === want) {
                return it;
            }
        }
        return null;
    }

    function importSource(path) {
        var found = findFootage(path);
        if (found) { return found; }
        var srcFile = new File(path);
        if (!srcFile.exists) { throw new Error("Source video not found:\n" + path); }
        return app.project.importFile(new ImportOptions(srcFile));
    }

    function ensureTemplate(M) {
        var comp = findComp("Caption Template");
        if (comp) {
            for (var i = 1; i <= comp.numLayers; i++) {
                if (comp.layer(i).name === "Caption Style") { return comp.layer(i); }
            }
            return comp.layer(1);
        }
        // first run: a sane static default the user then restyles + animates
        comp = app.project.items.addComp("Caption Template", M.out_w, M.out_h, 1, 5, M.fps);
        var t = comp.layers.addText("Caption Style");
        t.name = "Caption Style";
        var sp = t.property("Source Text");
        var td = sp.value;
        td.resetCharStyle();
        td.fontSize = 64;
        td.applyFill = true;
        td.fillColor = [1, 1, 1];
        td.applyStroke = true;
        td.strokeColor = [0, 0, 0];
        td.strokeWidth = 6;
        td.strokeOverFill = false;
        td.justification = ParagraphJustification.CENTER_JUSTIFY;
        sp.setValue(td);
        t.property("Transform").property("Position").setValue([M.out_w / 2, M.out_h - 70]);
        return t;
    }

    function ensureZoomTemplate(M) {
        // returns the Scale property whose keyframes define the punch shape (%).
        var comp = findComp("Zoom Template");
        if (comp) {
            var layer = null;
            for (var i = 1; i <= comp.numLayers; i++) {
                if (comp.layer(i).name === "Zoom Style") { layer = comp.layer(i); break; }
            }
            if (!layer) { layer = comp.layer(1); }
            return layer.property("Transform").property("Scale");
        }
        comp = app.project.items.addComp("Zoom Template", M.out_w, M.out_h, 1, 3, M.fps);
        var nul = comp.layers.addNull();
        nul.name = "Zoom Style";
        var sc = nul.property("Transform").property("Scale");
        // 1st segment = fixed punch-in, last segment = fixed punch-out, the middle
        // creep stretches to the clip. Defaults: 1s in, 1s out (see applyZoom).
        sc.setValueAtTime(0, [100, 100]);   // start
        sc.setValueAtTime(1, [190, 190]);   // punch-in done (1s)
        sc.setValueAtTime(2, [200, 200]);   // max -> creep start
        sc.setValueAtTime(3, [100, 100]);   // back out (1s)
        return sc;
    }

    function ensureSectionTemplate(M) {
        // The whole "Section Card" comp is the template: EVERY layer in it is cloned
        // onto each title card, and the "Section Style" text layer receives the title.
        // Restyle it / add layers (backgrounds, logos, accents) and they propagate.
        var comp = findComp("Section Card");
        if (comp) { return comp; }
        comp = app.project.items.addComp("Section Card", M.out_w, M.out_h, 1, CARD_S, M.fps);
        // styled default: dark full-frame backdrop + an accent bar behind the title
        comp.layers.addSolid([0.04, 0.04, 0.05], "Section BG", M.out_w, M.out_h, 1, CARD_S);
        var bar = comp.layers.addSolid([0.16, 0.55, 0.95], "Section Accent",
                                       M.out_w, 10, 1, CARD_S);
        bar.property("Transform").property("Position").setValue([M.out_w / 2, M.out_h / 2 + 80]);
        var t = comp.layers.addText("Section Style");
        t.name = "Section Style";
        var sp = t.property("Source Text");
        var td = sp.value;
        td.resetCharStyle();
        td.fontSize = 110;
        td.applyFill = true;
        td.fillColor = [1, 1, 1];
        td.applyStroke = true;
        td.strokeColor = [0, 0, 0];
        td.strokeWidth = 9;
        td.strokeOverFill = false;
        td.justification = ParagraphJustification.CENTER_JUSTIFY;
        sp.setValue(td);
        t.property("Transform").property("Position").setValue([M.out_w / 2, M.out_h / 2]);
        return comp;
    }

    // --- captions: clone template, stretch its animation to the caption ----

    function eachKeyedProp(g, fn) {
        for (var i = 1; i <= g.numProperties; i++) {
            var p = g.property(i);
            if (p.propertyType === PropertyType.PROPERTY) {
                if (p.canVaryOverTime && p.numKeys > 0) { fn(p); }
            } else {
                eachKeyedProp(p, fn);
            }
        }
    }

    function remapProp(p, t0, t1, lo, hi) {
        // AE has no setKeyTime: capture keys, remove them, re-add at scaled times.
        var n = p.numKeys, S = (t1 - t0) / (hi - lo), i;
        var times = [], vals = [], inI = [], outI = [], inE = [], outE = [];
        for (i = 1; i <= n; i++) {
            times.push(t0 + (p.keyTime(i) - lo) * S);
            vals.push(p.keyValue(i));
            inI.push(p.keyInInterpolationType(i));
            outI.push(p.keyOutInterpolationType(i));
            // ease only valid on BEZIER keys; capture defensively
            try { inE.push(p.keyInTemporalEase(i)); outE.push(p.keyOutTemporalEase(i)); }
            catch (e) { inE.push(null); outE.push(null); }
        }
        for (i = n; i >= 1; i--) { p.removeKey(i); }
        if (n === 1) { p.setValueAtTime(times[0], vals[0]); return; }
        p.setValuesAtTimes(times, vals);
        for (i = 1; i <= n; i++) {   // restore the authored feel
            try { p.setInterpolationTypeAtKey(i, inI[i - 1], outI[i - 1]); } catch (e) {}
            if (inE[i - 1]) {
                try { p.setTemporalEaseAtKey(i, inE[i - 1], outE[i - 1]); } catch (e) {}
            }
        }
    }

    function stretchKeys(layer, t0, t1) {
        var lo = Infinity, hi = -Infinity;
        eachKeyedProp(layer, function (p) {
            if (p.keyTime(1) < lo) { lo = p.keyTime(1); }
            if (p.keyTime(p.numKeys) > hi) { hi = p.keyTime(p.numKeys); }
        });
        if (lo === Infinity || hi <= lo) { return; }   // no/zero-span anim -> static
        eachKeyedProp(layer, function (p) { remapProp(p, t0, t1, lo, hi); });
    }

    function fadeOut(layer, end) {
        var op = layer.property("Transform").property("Opacity");
        var fade = Math.min(0.2, (end - layer.inPoint) * 0.5);
        if (fade <= 0) { return; }
        var t0 = end - fade;
        op.setValueAtTime(t0, op.valueAtTime(t0, false));
        op.setValueAtTime(end, 0);
        var ease = new KeyframeEase(0, 33.3333);   // easy ease the fade
        var n = op.numKeys;                        // our two fade keys are the last two
        op.setInterpolationTypeAtKey(n - 1, KeyframeInterpolationType.BEZIER,
                                            KeyframeInterpolationType.BEZIER);
        op.setInterpolationTypeAtKey(n, KeyframeInterpolationType.BEZIER,
                                        KeyframeInterpolationType.BEZIER);
        op.setTemporalEaseAtKey(n - 1, [ease], [ease]);
        op.setTemporalEaseAtKey(n, [ease], [ease]);
    }

    function footageLayer(comp) {
        for (var i = 1; i <= comp.numLayers; i++) {
            var L = comp.layer(i);
            if ((L instanceof AVLayer) && !(L instanceof TextLayer) &&
                L.source instanceof FootageItem) { return L; }
        }
        return null;
    }

    function _stampTemplate(zoomScale, w0, w1, valMul, out) {
        // Stamp the template punch into the window [w0, w1], appending keys to `out`
        // (parallel arrays). Punch-in (1st segment) + punch-out (last) keep their
        // authored durations anchored to w0/w1; the middle creep stretches to fill.
        var n = zoomScale.numKeys;
        var lo = zoomScale.keyTime(1), hi = zoomScale.keyTime(n), dur = w1 - w0;
        var inDur = (n >= 2) ? zoomScale.keyTime(2) - lo : 0;
        var outDur = (n >= 2) ? hi - zoomScale.keyTime(n - 1) : 0;
        var t2 = (n >= 2) ? zoomScale.keyTime(2) : lo;
        var tN1 = (n >= 2) ? zoomScale.keyTime(n - 1) : hi;
        var piecewise = (n >= 4 && tN1 > t2 && inDur + outDur < dur);
        function mapT(t) {
            if (!piecewise) { return w0 + ((hi > lo) ? (t - lo) / (hi - lo) * dur : 0); }
            if (t <= t2) { return w0 + (t - lo); }            // fixed punch-in
            if (t >= tN1) { return w1 - (hi - t); }           // fixed punch-out at end
            return w0 + inDur + (t - t2) / (tN1 - t2) * (dur - outDur - inDur);  // creep
        }
        for (var i = 1; i <= n; i++) {
            out.times.push(mapT(zoomScale.keyTime(i)));
            var s = valMul * zoomScale.keyValue(i)[0] / 100;  // uniform zoom %
            out.vals.push([s, s]);
            out.inI.push(zoomScale.keyInInterpolationType(i));
            out.outI.push(zoomScale.keyOutInterpolationType(i));
            try { out.inE.push(zoomScale.keyInTemporalEase(i)); out.outE.push(zoomScale.keyOutTemporalEase(i)); }
            catch (e) { out.inE.push(null); out.outE.push(null); }
        }
    }

    function applyZoom(zoomScale, footScale, dur, valMul, episodes) {
        // Stamp the template punch once per motion episode (clip-relative [start,end]
        // seconds from the manifest); between/around episodes the footage holds at the
        // 100% fit (valMul). No episodes -> static fit. Punches only on interesting
        // motion, can fire multiple times per clip.
        for (var j = footScale.numKeys; j >= 1; j--) { footScale.removeKey(j); }
        var n = zoomScale ? zoomScale.numKeys : 0;
        if (n === 0 || !episodes || episodes.length === 0) {
            footScale.setValue([valMul, valMul]); return;            // static fit
        }
        var out = { times: [], vals: [], inI: [], outI: [], inE: [], outE: [] }, k;
        for (k = 0; k < episodes.length; k++) {
            var w0 = Math.max(0, episodes[k].start), w1 = Math.min(dur, episodes[k].end);
            if (w1 > w0) { _stampTemplate(zoomScale, w0, w1, valMul, out); }
        }
        if (out.times.length === 0) { footScale.setValue([valMul, valMul]); return; }
        footScale.setValue([valMul, valMul]);   // baseline held outside the episodes
        footScale.setValuesAtTimes(out.times, out.vals);
        for (k = 1; k <= out.times.length; k++) {   // carry the punch easing over
            try { footScale.setInterpolationTypeAtKey(k, out.inI[k - 1], out.outI[k - 1]); } catch (e) {}
            if (out.inE[k - 1]) {
                try { footScale.setTemporalEaseAtKey(k, out.inE[k - 1], out.outE[k - 1]); } catch (e) {}
            }
        }
    }

    function buildCaption(comp, tmpl, cap) {
        tmpl.copyToComp(comp);                 // full clone, pastes as layer(1)
        var L = comp.layer(1);
        var sp = L.property("Source Text");
        var d = sp.value;
        d.text = cap.text;
        sp.setValue(d);                        // keep template char style
        L.inPoint = Math.max(0, cap.start);
        var animEnd = Math.min(comp.duration, cap.end);
        // linger ~10 frames past the last word so quick dialogue stays readable
        // and overlaps the next caption; the intro keeps its original timing.
        L.outPoint = Math.min(comp.duration, cap.end + 10 / comp.frameRate);
        stretchKeys(L, L.inPoint, animEnd);    // first kf -> start, last kf -> last word
        fadeOut(L, L.outPoint);
    }

    // --- build -------------------------------------------------------------

    function buildClip(M, clip, src, tmpl, zoomScale, idx) {
        var dur = clip.end - clip.start;
        var comp = app.project.items.addComp(
            "reFire clip " + idx, M.out_w, M.out_h, 1, dur, M.fps);

        var foot = comp.layers.add(src);
        foot.startTime = -clip.start;
        foot.inPoint = 0;
        foot.outPoint = dur;

        var tr = foot.property("Transform");
        tr.property("Anchor Point").setValue([0, src.height]);
        tr.property("Position").setValue([0, M.out_h]);
        applyZoom(zoomScale, tr.property("Scale"), dur,
                  M.out_w / src.width * 100, clip.zoom_episodes);

        var caps = clip.captions || [];
        for (var ci = 0; ci < caps.length; ci++) {
            buildCaption(comp, tmpl, caps[ci]);
        }
        return comp;
    }

    function populateCard(cardComp, dest, title) {
        // clone EVERY template layer onto dest (bottom-to-top keeps the stacking),
        // then set the title on the "Section Style" text layer.
        for (var i = dest.numLayers; i >= 1; i--) { dest.layer(i).remove(); }
        for (i = cardComp.numLayers; i >= 1; i--) { cardComp.layer(i).copyToComp(dest); }
        for (i = 1; i <= dest.numLayers; i++) {
            var L = dest.layer(i);
            if (L.name === "Section Style" && (L instanceof TextLayer)) {
                var sp = L.property("Source Text");
                var d = sp.value;
                d.text = title;
                sp.setValue(d);
                break;
            }
        }
    }

    function buildCard(M, title, cardComp) {
        var comp = app.project.items.addComp(
            "reFire card: " + title, M.out_w, M.out_h, 1, CARD_S, M.fps);
        populateCard(cardComp, comp, title);
        return comp;
    }

    function buildMaster(M, comps, cardTmpl) {
        // order the comps by section, dropping a title card before each titled
        // section, then lay them out with a short crossfade between neighbours.
        var sections = M.sections, s, ci, idx;
        if (!sections || !sections.length) {
            sections = [{ title: "", clip_indices: [] }];
            for (s = 0; s < comps.length; s++) { sections[0].clip_indices.push(s); }
        }
        var seq = [];
        for (s = 0; s < sections.length; s++) {
            if (sections[s].title) { seq.push(buildCard(M, sections[s].title, cardTmpl)); }
            var idxs = sections[s].clip_indices || [];
            for (ci = 0; ci < idxs.length; ci++) {
                idx = idxs[ci];
                if (idx >= 0 && idx < comps.length) { seq.push(comps[idx]); }
            }
        }
        if (!seq.length) { for (var q = 0; q < comps.length; q++) { seq.push(comps[q]); } }

        var i, xf = XFADE_S, minDur = seq[0].duration;
        for (i = 1; i < seq.length; i++) { if (seq[i].duration < minDur) { minDur = seq[i].duration; } }
        if (xf > minDur * 0.4) { xf = minDur * 0.4; }   // keep fades inside short comps
        var total = 0;
        for (i = 0; i < seq.length; i++) { total += seq[i].duration; }
        total -= xf * Math.max(0, seq.length - 1);

        var master = app.project.items.addComp(
            "reFire Master", M.out_w, M.out_h, 1, Math.max(total, 1), M.fps);
        var t = 0;
        for (i = 0; i < seq.length; i++) {
            var lay = master.layers.add(seq[i]);   // later layers stack on top -> fade in over prev
            lay.startTime = t;
            var op = lay.property("Transform").property("Opacity");
            var st = lay.startTime, en = st + seq[i].duration;
            if (i > 0 && xf > 0) { op.setValueAtTime(st, 0); op.setValueAtTime(st + xf, 100); }
            if (i < seq.length - 1 && xf > 0) { op.setValueAtTime(en - xf, 100); op.setValueAtTime(en, 0); }
            t = en - xf;
        }
        app.project.renderQueue.items.add(master);
    }

    function doBuild() {
        var M = readManifest();
        if (!M) { return "Pick a manifest first."; }
        app.beginUndoGroup("reFire build");
        try {
            var src = importSource(M.source);
            var tmpl = ensureTemplate(M);
            var zoomScale = ensureZoomTemplate(M);
            var cardTmpl = ensureSectionTemplate(M);
            var comps = [];
            for (var c = 0; c < M.clips.length; c++) {
                comps.push(buildClip(M, M.clips[c], src, tmpl, zoomScale, c));
            }
            buildMaster(M, comps, cardTmpl);
            return "Built " + comps.length + " clip(s). Master in render queue.";
        } finally {
            app.endUndoGroup();
        }
    }

    // --- update: refresh captions + zoom, keep footage ---------------------

    function doUpdate() {
        var M = readManifest();
        if (!M) { return "Pick a manifest first."; }
        if (!findComp("Caption Template")) { return "No template yet -- run Build first."; }
        var tmpl = ensureTemplate(M);
        var zoomScale = ensureZoomTemplate(M);
        var cardComp = ensureSectionTemplate(M);
        app.beginUndoGroup("reFire update");
        try {
            var touched = 0;
            for (var c = 0; c < M.clips.length; c++) {
                var comp = findComp("reFire clip " + c);
                if (!comp) { continue; }
                for (var i = comp.numLayers; i >= 1; i--) {   // drop old captions
                    if (comp.layer(i) instanceof TextLayer) { comp.layer(i).remove(); }
                }
                var foot = footageLayer(comp);   // re-punch zoom from the template
                if (foot) {
                    applyZoom(zoomScale, foot.property("Transform").property("Scale"),
                              comp.duration, M.out_w / foot.source.width * 100,
                              M.clips[c].zoom_episodes);
                }
                var caps = M.clips[c].captions || [];
                for (var ci = 0; ci < caps.length; ci++) {
                    buildCaption(comp, tmpl, caps[ci]);
                }
                touched++;
            }
            // re-clone the card template onto every existing section card so style +
            // any added layers propagate ("if not done so already").
            var cards = 0, secs = M.sections || [];
            for (var s = 0; s < secs.length; s++) {
                if (!secs[s].title) { continue; }
                var card = findComp("reFire card: " + secs[s].title);
                if (card) { populateCard(cardComp, card, secs[s].title); cards++; }
            }
            return touched ? "Updated " + touched + " clip(s) + " + cards + " card(s)."
                           : "No reFire clip comps found -- run Build first.";
        } finally {
            app.endUndoGroup();
        }
    }

    // --- generate: re-run the Python export (topic -> organize + manifest) --

    function doGenerate(topic, count, model, minScore, zoomSens) {
        // Re-runs `python -m refire ae` so the topic drives the Ollama grouping and
        // a fresh manifest (zoom_episodes + sections) is written, then reloads it.
        // Needs an existing manifest chosen first -- it supplies the video + run dir.
        var M = readManifest();
        if (!M) { return "Choose an existing manifest first (sets video + run dir)."; }
        var mf = new File(manifestPath);
        var runDir = mf.parent.parent;                 // <run>/ae/manifest.json -> <run>
        var cmd = 'cmd.exe /c python -m refire ae "' + M.source +
                  '" --run-dir "' + runDir.fsName + '"';
        if (topic && topic.length) { cmd += ' --topic "' + topic + '"'; }
        if (count && ("" + count).length) { cmd += ' --count ' + count; }
        if (model && model.length) { cmd += ' --model ' + model; }
        if (minScore && ("" + minScore).length) { cmd += ' --min-score ' + minScore; }
        if (zoomSens && ("" + zoomSens).length) { cmd += ' --zoom-sens ' + zoomSens; }
        cmd += ' 2>&1';
        var res = system.callSystem(cmd);
        var M2 = readManifest();                        // confirm it actually rewrote
        if (M2 && M2.clips && M2.clips.length &&
            M2.clips[0].zoom_episodes !== undefined) {
            var secs = M2.sections ? M2.sections.length : 0;
            return "Generated " + M2.clips.length + " clip(s), " + secs +
                   " section(s). Now click Build. " + (res || "");
        }
        return "Generate may have failed (no new manifest). Output:\n" + (res || "(none)");
    }

    // --- UI ----------------------------------------------------------------

    function buildUI(thisObj) {
        var w = (thisObj instanceof Panel)
            ? thisObj
            : new Window("palette", "reFire", undefined, { resizeable: true });
        w.orientation = "column";
        w.alignChildren = ["fill", "top"];
        w.spacing = 8;
        w.margins = 14;

        // minimalist "opencode" palette: dark canvas, mono text, one accent.
        // ponytail: native buttons can't be themed in ScriptUI -- left as-is.
        var INK = [0.85, 0.86, 0.88], MUTE = [0.55, 0.56, 0.60],
            ACCENT = [0.45, 0.85, 0.62], BG = [0.12, 0.12, 0.14];
        var texts = [], labels = [];
        function field(parent, label, def, chars) {       // "label  [____]" row
            var grp = parent.add("group");
            grp.spacing = 8;
            var l = grp.add("statictext", undefined, label); labels.push(l);
            var e = grp.add("edittext", undefined, def || "");
            e.characters = chars; texts.push(e);
            return e;
        }
        function panel(title) {
            var p = w.add("panel", undefined, title);
            p.orientation = "column"; p.alignChildren = ["fill", "top"];
            p.margins = 10; p.spacing = 6; labels.push(p);
            return p;
        }

        var srcP = panel("Source");
        var pathTxt = srcP.add("statictext", undefined, "No manifest selected");
        pathTxt.minimumSize = [240, 20]; texts.push(pathTxt);
        var btnPick = srcP.add("button", undefined, "Choose manifest…");

        var genP = panel("Generate");
        var topicTxt = field(genP, "Topic", "", 22);
        var countTxt = field(genP, "Count", "", 6);
        var modelTxt = field(genP, "Model", "llama3.1:8b", 16);
        var scoreTxt = field(genP, "Min score", "", 5);   // cut boring clips, 0-10
        var zoomTxt = field(genP, "Zoom amt", "1.0", 5);   // 1=normal, >1 more
        var btnGen = genP.add("button", undefined, "Generate (topic → organize)");

        var buildP = panel("Build");
        var btnBuild = buildP.add("button", undefined, "Build");
        var btnUpd = buildP.add("button", undefined, "Update");

        var status = w.add("statictext", undefined, "", { multiline: true });
        status.minimumSize = [240, 48];

        // apply the palette + monospace; cosmetic only, never break the panel
        try {
            var g = w.graphics, mono = ScriptUI.newFont("Consolas", "Regular", 12);
            w.graphics.backgroundColor = g.newBrush(g.BrushType.SOLID_COLOR, BG);
            function paint(c, rgb) {
                c.graphics.font = mono;
                c.graphics.foregroundColor = g.newPen(g.PenType.SOLID_COLOR, rgb, 1);
            }
            for (var i = 0; i < labels.length; i++) paint(labels[i], MUTE);
            for (var j = 0; j < texts.length; j++) paint(texts[j], INK);
            paint(status, ACCENT);
        } catch (e) { /* older AE / no Consolas -> default chrome, still works */ }

        btnPick.onClick = function () {
            var f = File.openDialog("Select reFire manifest.json", "*.json");
            if (f) {
                manifestPath = f.fsName;
                pathTxt.text = decodeURI(f.name);
                status.text = "";
            }
        };
        btnGen.onClick = function () {
            status.text = "Generating… (running detector export)";
            try {
                status.text = doGenerate(topicTxt.text, countTxt.text, modelTxt.text,
                                         scoreTxt.text, zoomTxt.text);
            }
            catch (e) { status.text = "Error: " + e.toString(); }
        };
        btnBuild.onClick = function () {
            try { status.text = doBuild(); }
            catch (e) { status.text = "Error: " + e.toString(); }
        };
        btnUpd.onClick = function () {
            try { status.text = doUpdate(); }
            catch (e) { status.text = "Error: " + e.toString(); }
        };

        w.layout.layout(true);
        return w;
    }

    var ui = buildUI(thisObj);
    if (ui instanceof Window) { ui.center(); ui.show(); }
})(this);
