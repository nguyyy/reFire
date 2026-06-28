// reFire — After Effects caption panel (ExtendScript, ES3).
//
// Launch as a dockable panel (drop in AE's "ScriptUI Panels" folder, then
// Window > reFire) or as a floating palette (File > Scripts > Run Script File...).
//
//   Make (one click)    the hands-off flow: pick an Output folder, type a VOD #,
//                        a free-text Brief (what video you want), and a target
//                        Duration (e.g. 20m). Shells `python -m refire make` ->
//                        downloads the VOD, transcribes/embeds (cached), retrieves
//                        + LLM-scores clips against the brief, packs them to the
//                        duration budget chronologically, writes the manifest, and
//                        auto-loads it. Tuning row: zoom amount / words-per-line /
//                        duration tolerance. Needs python + ollama on PATH. Then Build.
//   Load manifest...    (alt) point Build at an existing run/ae/manifest.json
//                        instead of running Make.
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
                // never retime Time Remap: it's driven by loopOut on overlays, and
                // stretching it would re-speed the emote instead of the punch.
                if (p.matchName === "ADBE Time Remapping") { continue; }
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
        // Restore the authored feel. Order matters: set the captured temporal ease
        // FIRST (this also makes the key Bezier with the template's influence), then
        // set interpolation type ONLY for non-Bezier keys. Re-stamping BEZIER would
        // reset the ease to AE's default Easy Ease -- the bug this fixes.
        for (i = 1; i <= n; i++) {
            if (inE[i - 1] && outE[i - 1]) {
                try { p.setTemporalEaseAtKey(i, inE[i - 1], outE[i - 1]); } catch (e) {}
            }
        }
        for (i = 1; i <= n; i++) {
            if (inI[i - 1] !== KeyframeInterpolationType.BEZIER ||
                outI[i - 1] !== KeyframeInterpolationType.BEZIER) {
                try { p.setInterpolationTypeAtKey(i, inI[i - 1], outI[i - 1]); } catch (e) {}
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
        // linger ~10 frames past the last word so quick dialogue stays readable
        // and overlaps the next caption; the intro keeps its original timing.
        L.outPoint = Math.min(comp.duration, cap.end + 10 / comp.frameRate);
        stretchKeys(L, L.inPoint, L.outPoint);  // first kf -> start, last kf -> out
        fadeOut(L, L.outPoint);
    }

    // --- overlays: emote punch-in + impact SFX on funny moments ------------

    function ensureOverlayTemplate(M) {
        // returns the layer whose Transform keyframes define the impact punch-in; the
        // emote image is swapped onto a clone of it per overlay, keeping the animation.
        // Restyle/move/re-time THIS one layer and Update to re-punch every overlay.
        var comp = findComp("Overlay Template");
        if (comp) {
            for (var i = 1; i <= comp.numLayers; i++) {
                if (comp.layer(i).name === "Overlay Style") { return comp.layer(i); }
            }
            return comp.layer(1);
        }
        comp = app.project.items.addComp("Overlay Template", M.out_w, M.out_h, 1, 3.0, M.fps);
        var sol = comp.layers.addSolid([0.95, 0.85, 0.20], "Overlay Style", 200, 200, 1);
        sol.name = "Overlay Style";
        var tr = sol.property("Transform");
        tr.property("Position").setValue([M.out_w / 2, M.out_h * 0.40]);
        var sc = tr.property("Scale");          // bounce in, hold, pop out (% of emote px)
        sc.setValueAtTime(0.00, [0, 0]);
        sc.setValueAtTime(0.18, [360, 360]);    // overshoot
        sc.setValueAtTime(0.34, [300, 300]);    // settle
        sc.setValueAtTime(2.70, [300, 300]);    // hold
        sc.setValueAtTime(3.00, [0, 0]);        // pop out
        var op = tr.property("Opacity");
        op.setValueAtTime(0.00, 0);
        op.setValueAtTime(0.12, 100);
        return sol;
    }

    function addSfx(comp, path, atTime, folder) {
        var snd = importSource(path);            // wav/mp3 via the same importer/dedupe
        if (folder) { snd.parentFolder = folder; }
        var L = comp.layers.add(snd);
        L.name = "reFire sfx";
        L.startTime = atTime;
        L.inPoint = atTime;
        var natural = atTime + (snd.duration || 3.5);
        L.outPoint = Math.min(comp.duration, natural);
    }

    function buildOverlay(comp, tmpl, ov, folder) {
        if (!ov || !ov.asset) { return; }
        var emote = importSource(ov.asset);      // png/gif punch-in image
        if (folder) { emote.parentFolder = folder; }
        tmpl.copyToComp(comp);                   // clone the animated placeholder -> layer(1)
        var L = comp.layer(1);
        L.replaceSource(emote, false);           // swap the emote in, keep the keyframes
        L.name = "reFire overlay";
        // re-center the anchor on the emote (the placeholder solid was a different size)
        L.property("Transform").property("Anchor Point").setValue([emote.width / 2, emote.height / 2]);

        var dur = ov.duration || 3;
        var inP = Math.max(0, ov.start);
        var outP = Math.min(comp.duration, inP + dur);
        // An animated GIF imports as short, finite footage -- left alone its layer
        // out point clamps to a frame or two and the whole punch collapses to one
        // frame. Time Remap frees the duration; loopOut keeps the emote moving for
        // the full hold. Stills (png/jpg, duration 0) already stretch to any length.
        // ponytail: loopOut('cycle') replays the gif; use 'none' to hold frame 1.
        if (emote.duration && emote.duration > 0) {
            try {
                L.timeRemapEnabled = true;
                L.property("Time Remap").expression = "loopOut('cycle')";
            } catch (e) {}
        }
        L.startTime = inP;
        L.inPoint = inP;
        L.outPoint = outP;
        stretchKeys(L, inP, outP);               // first kf -> start, last kf -> out
        if (ov.sfx) { addSfx(comp, ov.sfx, inP, folder); }
    }

    // --- build -------------------------------------------------------------

    function buildClip(M, clip, src, tmpl, zoomScale, idx, folder, overlayTmpl, ovFolder) {
        var dur = clip.end - clip.start;
        var comp = app.project.items.addComp(
            "reFire clip " + idx, M.out_w, M.out_h, 1, dur, M.fps);
        if (folder) { comp.parentFolder = folder; }

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
        // overlays last so the emote sits on top of the captions
        var ovs = clip.overlays || [];
        for (var oi = 0; oi < ovs.length; oi++) {
            buildOverlay(comp, overlayTmpl, ovs[oi], ovFolder);
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

    function buildCard(M, title, cardComp, folder) {
        var comp = app.project.items.addComp(
            "reFire card: " + title, M.out_w, M.out_h, 1, CARD_S, M.fps);
        if (folder) { comp.parentFolder = folder; }
        populateCard(cardComp, comp, title);
        return comp;
    }

    function buildMaster(M, comps, cardTmpl, folder) {
        // order the comps by section, dropping a title card before each titled
        // section, then lay them out with a short crossfade between neighbours.
        var sections = M.sections, s, ci, idx;
        if (!sections || !sections.length) {
            sections = [{ title: "", clip_indices: [] }];
            for (s = 0; s < comps.length; s++) { sections[0].clip_indices.push(s); }
        }
        var seq = [];
        for (s = 0; s < sections.length; s++) {
            if (sections[s].title) { seq.push(buildCard(M, sections[s].title, cardTmpl, folder)); }
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
        if (folder) { master.parentFolder = folder; }
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
        // music bed under the whole cut: place once, fade in/out, hold at bgm_db.
        // ponytail: no loop -- a track shorter than the cut just ends; loop it manually
        // (enable Time Remap + loopOut) if you need it to fill.
        if (M.bgm) {
            try {
                var music = importSource(M.bgm);
                if (folder) { music.parentFolder = folder; }
                var mL = master.layers.add(music);
                mL.name = "reFire music";
                mL.startTime = 0;
                if (mL.outPoint > master.duration) { mL.outPoint = master.duration; }
                var lvl = mL.property("Audio").property("Audio Levels");
                var db = (M.bgm_db != null) ? M.bgm_db : -18;
                var fade = Math.min(1.0, master.duration * 0.45);
                lvl.setValueAtTime(0, [-48, -48]);
                lvl.setValueAtTime(fade, [db, db]);
                lvl.setValueAtTime(master.duration - fade, [db, db]);
                lvl.setValueAtTime(master.duration, [-48, -48]);
            } catch (e) { /* missing/odd audio -> silent cut, build still succeeds */ }
        }
        app.project.renderQueue.items.add(master);
    }

    function ensureFolder(name) {
        // reuse a shared folder (templates/footage) across builds; create on demand.
        for (var i = 1; i <= app.project.numItems; i++) {
            var it = app.project.item(i);
            if (it instanceof FolderItem && it.name === name) { return it; }
        }
        return app.project.items.addFolder(name);
    }

    function nextBuildFolder() {
        // a fresh masterbuild1 / masterbuild2 / ... so repeated Builds don't mingle.
        var n = 1;
        for (;;) {
            var nm = "masterbuild" + n, taken = false;
            for (var i = 1; i <= app.project.numItems; i++) {
                var it = app.project.item(i);
                if (it instanceof FolderItem && it.name === nm) { taken = true; break; }
            }
            if (!taken) { return app.project.items.addFolder(nm); }
            n++;
        }
    }

    function doBuild(report) {
        report = report || function () {};
        var M = readManifest();
        if (!M) { return "Pick a manifest first."; }
        app.beginUndoGroup("reFire build");
        try {
            var tplF = ensureFolder("reFire templates");   // shared singletons
            var footF = ensureFolder("reFire footage");
            var buildF = nextBuildFolder();                // this build's home

            var ovF = ensureFolder("reFire overlays");      // shared emote/sfx/music assets
            var src = importSource(M.source);
            src.parentFolder = footF;
            var tmpl = ensureTemplate(M);
            var zoomScale = ensureZoomTemplate(M);
            var cardTmpl = ensureSectionTemplate(M);
            var overlayTmpl = ensureOverlayTemplate(M);
            // tuck the (shared) templates into their folder
            var capC = findComp("Caption Template"); if (capC) { capC.parentFolder = tplF; }
            var zoomC = findComp("Zoom Template");   if (zoomC) { zoomC.parentFolder = tplF; }
            var ovC = findComp("Overlay Template");  if (ovC) { ovC.parentFolder = tplF; }
            if (cardTmpl) { cardTmpl.parentFolder = tplF; }

            var comps = [], N = M.clips.length;
            for (var c = 0; c < N; c++) {
                comps.push(buildClip(M, M.clips[c], src, tmpl, zoomScale, c, buildF,
                                     overlayTmpl, ovF));
                report(Math.round((c + 1) / N * 95), "building clip " + (c + 1) + "/" + N);
            }
            buildMaster(M, comps, cardTmpl, buildF);
            report(100, "master queued");
            return "Built " + comps.length + " clip(s) in '" + buildF.name +
                   "'. Master in render queue.";
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
        var overlayTmpl = ensureOverlayTemplate(M);
        var ovF = ensureFolder("reFire overlays");
        app.beginUndoGroup("reFire update");
        try {
            var touched = 0;
            for (var c = 0; c < M.clips.length; c++) {
                var comp = findComp("reFire clip " + c);
                if (!comp) { continue; }
                for (var i = comp.numLayers; i >= 1; i--) {   // drop old captions + overlays
                    var nm = comp.layer(i).name;
                    if (comp.layer(i) instanceof TextLayer ||
                        nm === "reFire overlay" || nm === "reFire sfx") {
                        comp.layer(i).remove();
                    }
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
                var ovs = M.clips[c].overlays || [];
                for (var oi = 0; oi < ovs.length; oi++) {
                    buildOverlay(comp, overlayTmpl, ovs[oi], ovF);
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

    // --- make: the hands-off pipeline (VOD# + brief + duration -> manifest) --

    function writeFile(f, text) { f.open("w"); f.write(text); f.close(); }

    function doMake(o) {
        // Launches `python -m refire make` in a VISIBLE, detached terminal so its live
        // progress (printed by the CLI) can be monitored, and the AE panel never freezes
        // (no blocking poll loop). A .bat carries the command, so there are no nested
        // quotes on the cmd line for cmd.exe to mangle. When it finishes, click
        // 'Load manifest...'. Needs python + ollama on PATH.
        if (!o.out) { return "Pick an output folder first."; }
        if (!o.vod || !o.vod.length) { return "Enter a VOD number."; }
        if (!o.brief || !o.brief.length) { return "Describe the video you want (Brief)."; }
        if (!o.duration || !o.duration.length) { return "Set a target Duration (e.g. 20m)."; }

        var runDir = o.out + "/run", vods = o.out + "/vods";
        new Folder(runDir).create();
        var batF = new File(runDir + "/reFire_make.bat");

        var brief = o.brief.replace(/"/g, "").replace(/[\r\n]+/g, " ");
        var py = 'python -m refire make ' + o.vod +
                 ' --brief "' + brief + '"' +
                 ' --duration ' + o.duration +
                 ' --run-dir "' + runDir + '"' +
                 ' --cache-dir "' + vods + '"';
        if (o.game && o.game.length) { py += ' --game "' + o.game.replace(/"/g, "") + '"'; }
        if (o.model && o.model.length) { py += ' --model ' + o.model; }
        if (o.zoom && o.zoom.length) { py += ' --zoom-sens ' + o.zoom; }
        if (o.wpl && o.wpl.length) { py += ' --words-per-line ' + o.wpl; }
        if (o.tol && o.tol.length) { py += ' --tol ' + o.tol; }

        writeFile(batF,
            "@echo off\r\n" +
            "title reFire make " + o.vod + "\r\n" +
            "echo reFire make -- live progress below. Leave this window open.\r\n" +
            "echo.\r\n" +
            py + "\r\n" +
            "echo.\r\n" +
            "echo ==== finished (exit %ERRORLEVEL%). Manifest: run\\ae\\manifest.json ====\r\n" +
            "echo Back in After Effects, click 'Load manifest...' then Build.\r\n" +
            "pause\r\n");

        // visible + detached: callSystem returns immediately, the terminal shows progress
        system.callSystem('cmd.exe /c start "reFire make" "' + batF.fsName + '"');
        return "Make launched in a terminal -- watch progress there.\n" +
               "When it finishes, click 'Load manifest...' (run/ae/manifest.json).";
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

        // "opencode" palette: dark canvas, mono text, one accent, box-rule dividers.
        // ponytail: native buttons/panel chrome can't be themed in ScriptUI -- the
        // text, fields, header and dividers carry the look.
        var INK = [0.86, 0.87, 0.90], MUTE = [0.52, 0.54, 0.60],
            ACCENT = [0.42, 0.86, 0.62], BG = [0.11, 0.11, 0.13];
        var LABELW = 78;
        var texts = [], labels = [], heads = [], rules = [];
        var outFolder = null;

        function rule(parent) {                            // a thin terminal-style divider
            var r = parent.add("statictext", undefined, mk("─", 46));
            r.alignment = ["fill", "top"]; rules.push(r); return r;
        }
        function mk(ch, n) { var s = ""; while (n-- > 0) { s += ch; } return s; }
        function panel(title) {
            var p = w.add("panel", undefined, title);
            p.orientation = "column"; p.alignChildren = ["fill", "top"];
            p.margins = [12, 12, 12, 12]; p.spacing = 6; labels.push(p);
            return p;
        }
        function field(parent, label, def, opts) {         // aligned "label [____]" row
            opts = opts || {};
            var grp = parent.add("group");
            grp.orientation = "row"; grp.spacing = 8;
            grp.alignChildren = ["left", "center"]; grp.alignment = ["fill", "top"];
            var l = grp.add("statictext", undefined, label);
            l.preferredSize = [LABELW, -1]; labels.push(l);
            var e;
            if (opts.multiline) {
                e = grp.add("edittext", undefined, def || "", { multiline: true });
                e.preferredSize = [-1, opts.h || 50]; e.alignment = ["fill", "center"];
            } else {
                e = grp.add("edittext", undefined, def || "");
                if (opts.chars) { e.characters = opts.chars; }
                e.alignment = opts.fill ? ["fill", "center"] : ["left", "center"];
            }
            texts.push(e); return e;
        }

        // header
        var head = w.add("statictext", undefined, "reFire"); heads.push(head);
        var tag = w.add("statictext", undefined, "vod → brief → edited cut");
        labels.push(tag);
        rule(w);

        // MAKE: the autonomous flow
        var makeP = panel("Make");
        var outGrp = makeP.add("group"); outGrp.orientation = "row"; outGrp.spacing = 8;
        outGrp.alignChildren = ["left", "center"]; outGrp.alignment = ["fill", "top"];
        var outLbl = outGrp.add("statictext", undefined, "Output");
        outLbl.preferredSize = [LABELW, -1]; labels.push(outLbl);
        var outTxt = outGrp.add("statictext", undefined, "(no folder)");
        outTxt.alignment = ["fill", "center"]; texts.push(outTxt);
        var btnOut = outGrp.add("button", undefined, "Pick…"); btnOut.preferredSize = [56, -1];

        var vodTxt = field(makeP, "VOD #", "", { chars: 14 });
        var briefTxt = field(makeP, "Brief", "", { multiline: true, h: 46 });
        var durTxt = field(makeP, "Duration", "20m", { chars: 10 });
        var gameTxt = field(makeP, "Game", "", { fill: true });
        var modelTxt = field(makeP, "Model", "llama3.1:8b", { fill: true });

        // advanced knobs on one compact row
        var advGrp = makeP.add("group"); advGrp.orientation = "row"; advGrp.spacing = 8;
        advGrp.alignChildren = ["left", "center"]; advGrp.alignment = ["fill", "top"];
        var advLbl = advGrp.add("statictext", undefined, "Tuning");
        advLbl.preferredSize = [LABELW, -1]; labels.push(advLbl);
        function mini(label, def, chars) {
            var l = advGrp.add("statictext", undefined, label); labels.push(l);
            var e = advGrp.add("edittext", undefined, def); e.characters = chars;
            texts.push(e); return e;
        }
        var zoomTxt = mini("zoom", "1.0", 4);
        var wplTxt = mini("words", "3", 3);
        var tolTxt = mini("tol", "0.25", 5);

        var btnMake = makeP.add("button", undefined, "Make");
        var note = makeP.add("statictext", undefined,
            "runs in the background; the bar tracks progress", { multiline: true });
        note.alignment = ["fill", "top"]; labels.push(note);

        rule(w);

        // BUILD: turn the loaded manifest into AE comps
        var buildP = panel("Build");
        var pathTxt = buildP.add("statictext", undefined, "No manifest loaded");
        pathTxt.alignment = ["fill", "top"]; texts.push(pathTxt);
        var btnPick = buildP.add("button", undefined, "Load manifest…");
        var rowB = buildP.add("group"); rowB.orientation = "row"; rowB.spacing = 8;
        rowB.alignment = ["fill", "top"];
        var btnBuild = rowB.add("button", undefined, "Build"); btnBuild.alignment = ["fill", "center"];
        var btnUpd = rowB.add("button", undefined, "Update"); btnUpd.alignment = ["fill", "center"];

        rule(w);
        var bar = w.add("progressbar", undefined, 0, 100);
        bar.preferredSize = [-1, 8]; bar.alignment = ["fill", "top"];
        var status = w.add("statictext", undefined, "ready", { multiline: true });
        status.minimumSize = [240, 56]; status.alignment = ["fill", "top"];

        // paint the palette + monospace; cosmetic only, wrapped so it never breaks
        try {
            var g = w.graphics;
            var mono = ScriptUI.newFont("Consolas", "Regular", 12);
            var monoH = ScriptUI.newFont("Consolas", ScriptUI.FontStyle.BOLD, 18);
            w.graphics.backgroundColor = g.newBrush(g.BrushType.SOLID_COLOR, BG);
            function paint(c, rgb, font) {
                c.graphics.font = font || mono;
                c.graphics.foregroundColor = g.newPen(g.PenType.SOLID_COLOR, rgb, 1);
            }
            var i;
            for (i = 0; i < labels.length; i++) { paint(labels[i], MUTE); }
            for (i = 0; i < texts.length; i++) { paint(texts[i], INK); }
            for (i = 0; i < rules.length; i++) { paint(rules[i], [0.24, 0.25, 0.29]); }
            for (i = 0; i < heads.length; i++) { paint(heads[i], ACCENT, monoH); }
            paint(status, ACCENT);
        } catch (e) { /* older AE / no Consolas -> default chrome, still works */ }

        function gather() {
            return { out: outFolder, vod: vodTxt.text, brief: briefTxt.text,
                     duration: durTxt.text, game: gameTxt.text, model: modelTxt.text,
                     zoom: zoomTxt.text, wpl: wplTxt.text, tol: tolTxt.text };
        }
        function report(pct, msg) {                 // drive bar + status, repaint live
            try {
                if (pct != null) { bar.value = pct; }
                if (msg != null) { status.text = (pct != null ? pct + "%  " : "") + msg; }
                w.update();
            } catch (e) {}
        }

        btnOut.onClick = function () {
            var f = Folder.selectDialog("Choose an output folder for reFire");
            if (f) { outFolder = f.fsName; outTxt.text = decodeURI(f.name); status.text = "ready"; }
        };
        btnMake.onClick = function () {
            try { status.text = doMake(gather()); }
            catch (e) { status.text = "Error: " + e.toString(); }
        };
        btnPick.onClick = function () {
            // if Make ran, the manifest is at <out>/run/ae/manifest.json -- auto-load it,
            // else open a file dialog (start near the output folder for convenience).
            if (outFolder) {
                var auto = new File(outFolder + "/run/ae/manifest.json");
                if (auto.exists) {
                    manifestPath = auto.fsName; pathTxt.text = "run/ae/manifest.json";
                    status.text = "manifest loaded -> click Build"; return;
                }
            }
            var start = outFolder ? new File(outFolder + "/run/ae/manifest.json") : null;
            var f = start ? start.openDlg("Select reFire manifest.json", "*.json")
                          : File.openDialog("Select reFire manifest.json", "*.json");
            if (f) { manifestPath = f.fsName; pathTxt.text = decodeURI(f.name); status.text = "ready"; }
        };
        btnBuild.onClick = function () {
            report(0, "building…");
            try { status.text = doBuild(report); } catch (e) { status.text = "Error: " + e.toString(); }
        };
        btnUpd.onClick = function () {
            try { status.text = doUpdate(); } catch (e) { status.text = "Error: " + e.toString(); }
        };

        w.layout.layout(true);
        return w;
    }

    var ui = buildUI(thisObj);
    if (ui instanceof Window) { ui.center(); ui.show(); }
})(this);
