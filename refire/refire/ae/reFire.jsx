// reFire â€” After Effects automation (ExtendScript, ES3).
//
// This file is the AE half of the reFire CEP panel; the UI lives next door in
// index.html and calls in here via evalScript. Install the panel with
// `refire/ae/install.ps1`, then Window > Extensions > reFire.
//
// API (everything else in here is private):
//   reFire.build(manifestPath)    import footage + build one comp per clip (footage
//                                 trim, bottom-left zoom keyframes, captions, emote
//                                 overlays + sfx) + a Master comp, queued to render.
//   reFire.shift(manifestPath, s) slide ONLY the captions by s seconds (+ = later,
//                                 - = earlier). Fast enough to nudge live when the
//                                 transcript's word timings read off.
//   reFire.update(manifestPath)   re-apply captions, zoom, overlays and section cards
//                                 on the already-built comps from the current
//                                 templates and manifest, without re-importing
//                                 footage. Use after tweaking a template or
//                                 re-exporting the manifest; new source needs build.
// Both return a human-readable status string, which the panel shows and logs.
//
// STYLE + ANIMATION: every caption is a clone of the text layer named
// "Caption Style" in the "Caption Template" comp. Restyle AND animate that ONE
// layer (font/colour/stroke + text animators / opacity keyframes), then click
// Update -- the look and the motion propagate to every caption. The template
// animation is stretched to fit each caption (first keyframe -> first word, last
// keyframe -> last word), plus a short fade-out at the last word.
//
// Each clip's captions are precomposed into a nested comp named "cc", so the whole
// caption track is one layer: drag it in the timeline (or use the panel's offset /
// reFire.shift) to slide every caption in that clip against the audio.
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

var reFire = (function () {
    var CARD_S = 1.5;          // section title-card duration (seconds)
    var XFADE_S = 0.25;        // crossfade between master layers (seconds)

    // --- manifest ----------------------------------------------------------

    function readManifest(manifestPath) {
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

    function templateLayer(comp, name) {
        // The named style layer, else the topmost one. NULL if the comp is empty --
        // AE throws "unable to call 'layer' ... the range has no elements" on
        // comp.layer(1) with zero layers, so every caller must re-stamp its default
        // instead of assuming a comp it found is still populated.
        if (!comp || comp.numLayers === 0) { return null; }
        for (var i = 1; i <= comp.numLayers; i++) {
            if (comp.layer(i).name === name) { return comp.layer(i); }
        }
        return comp.layer(1);
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
        var found = templateLayer(comp, "Caption Style");
        if (found) { return found; }
        // no comp yet, or its contents were deleted -> stamp a sane static default
        // back in (into the existing comp if there is one) for the user to restyle.
        if (!comp) {
            comp = app.project.items.addComp("Caption Template", M.out_w, M.out_h, 1, 5, M.fps);
        }
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
        var found = templateLayer(comp, "Zoom Style");
        if (found) { return found.property("Transform").property("Scale"); }
        if (!comp) {
            comp = app.project.items.addComp("Zoom Template", M.out_w, M.out_h, 1, 3, M.fps);
        }
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
        if (comp && comp.numLayers > 0) { return comp; }
        // an emptied card comp would clone nothing onto every title card -> re-stamp
        if (!comp) {
            comp = app.project.items.addComp("Section Card", M.out_w, M.out_h, 1, CARD_S, M.fps);
        }
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

    var FOOT_NAME = "reFire footage";

    function footageLayer(comp) {
        // Name first: past the 3h mark the footage layer's source is a shim COMP, not a
        // FootageItem, and an overlay emote (a FootageItem, stacked on top) would win the
        // structural check below. That check stays for comps built before the rename.
        for (var i = 1; i <= comp.numLayers; i++) {
            if (comp.layer(i).name === FOOT_NAME) { return comp.layer(i); }
        }
        for (i = 1; i <= comp.numLayers; i++) {
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

    // `off` shifts every caption in time (seconds, +late/-early) to correct word
    // timestamps that read a touch ahead of or behind the audio. It is ABSOLUTE, not
    // cumulative: captions are always re-stamped from the manifest, so nudging back
    // and forth never drifts.
    function buildCaption(comp, tmpl, cap, off) {
        off = Number(off) || 0;
        var inP = Math.max(0, cap.start + off);
        // linger ~10 frames past the last word so quick dialogue stays readable
        // and overlaps the next caption; the intro keeps its original timing.
        var outP = Math.min(comp.duration, cap.end + off + 10 / comp.frameRate);
        if (outP <= inP) { return; }           // shifted clean off this clip
        tmpl.copyToComp(comp);                 // full clone, pastes as layer(1)
        var L = comp.layer(1);
        var sp = L.property("Source Text");
        var d = sp.value;
        d.text = cap.text;
        sp.setValue(d);                        // keep template char style
        L.inPoint = inP;
        L.outPoint = outP;
        stretchKeys(L, L.inPoint, L.outPoint);  // first kf -> start, last kf -> out
        fadeOut(L, L.outPoint);
    }

    var CC_NAME = "cc";

    // Every clip's captions get precomposed into their own comp named "cc". The text
    // is stamped at the manifest times inside it, so the offset is just the "cc"
    // layer's startTime -- one layer to slide instead of a full re-stamp, and the
    // caption track stays draggable by hand in the timeline.
    function buildCaptions(M, comp, tmpl, caps, capOff, folder) {
        if (!caps || !caps.length) { return; }
        var cc = app.project.items.addComp(CC_NAME, M.out_w, M.out_h, 1,
                                           comp.duration, M.fps);
        if (folder) { cc.parentFolder = folder; }
        for (var ci = 0; ci < caps.length; ci++) { buildCaption(cc, tmpl, caps[ci], 0); }
        var L = comp.layers.add(cc);
        L.name = CC_NAME;
        shiftCaptions(L, capOff);
    }

    // `off` is ABSOLUTE (seconds, +late/-early): it is the layer's startTime, so
    // nudging back and forth never drifts. In/out clamp to the clip -- captions
    // pushed off either end are simply not shown.
    function shiftCaptions(L, off) {
        off = Number(off) || 0;
        var dur = L.containingComp.duration;
        var inP = Math.max(0, off), outP = Math.min(dur, off + L.source.duration);
        if (outP <= inP) { L.enabled = false; return; }   // shifted clean off the clip
        L.enabled = true;
        L.startTime = off;
        L.inPoint = inP;
        L.outPoint = outP;
    }

    function captionLayer(comp) {
        for (var i = 1; i <= comp.numLayers; i++) {
            if (comp.layer(i).name === CC_NAME) { return comp.layer(i); }
        }
        return null;
    }

    function dropCaptions(comp) {
        // takes the nested comp with it, else the project fills with orphan "cc"s.
        // The TextLayer sweep clears captions from comps built before precomping.
        for (var i = comp.numLayers; i >= 1; i--) {
            var L = comp.layer(i);
            if (L.name === CC_NAME) { var src = L.source; L.remove(); if (src) { src.remove(); } }
            else if (L instanceof TextLayer) { L.remove(); }
        }
    }

    // --- overlays: emote punch-in + impact SFX on funny moments ------------

    function ensureOverlayTemplate(M) {
        // returns the layer whose Transform keyframes define the impact punch-in; the
        // emote image is swapped onto a clone of it per overlay, keeping the animation.
        // Restyle/move/re-time THIS one layer and Update to re-punch every overlay.
        var comp = findComp("Overlay Template");
        var found = templateLayer(comp, "Overlay Style");
        if (found) { return found; }
        if (!comp) {
            comp = app.project.items.addComp("Overlay Template", M.out_w, M.out_h, 1, 3.0, M.fps);
        }
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

    var XFER = ["Anchor Point", "Position", "Scale", "Rotation", "Opacity"];

    function copyTransform(from, to) {
        // Carry the template's Transform animation onto another layer. Used when the
        // template has no source to replace (see buildOverlay).
        var f = from.property("Transform"), t = to.property("Transform"), i, k;
        for (i = 0; i < XFER.length; i++) {
            var src = f.property(XFER[i]), dst = t.property(XFER[i]);
            if (!src || !dst) { continue; }
            for (k = dst.numKeys; k >= 1; k--) { dst.removeKey(k); }
            if (src.numKeys === 0) { try { dst.setValue(src.value); } catch (e) {} continue; }
            var times = [], vals = [];
            for (k = 1; k <= src.numKeys; k++) { times.push(src.keyTime(k)); vals.push(src.keyValue(k)); }
            if (times.length === 1) { dst.setValueAtTime(times[0], vals[0]); }
            else { dst.setValuesAtTimes(times, vals); }
            for (k = 1; k <= src.numKeys; k++) {   // keep the authored feel (see remapProp)
                try { dst.setTemporalEaseAtKey(k, src.keyInTemporalEase(k), src.keyOutTemporalEase(k)); } catch (e) {}
                try { dst.setInterpolationTypeAtKey(k, src.keyInInterpolationType(k),
                                                       src.keyOutInterpolationType(k)); } catch (e) {}
            }
        }
    }

    function hasSource(L) {
        // shape/text/null/adjustment layers (and cameras/lights) have no replaceable
        // source; AE throws "layer does not have a source" on the property itself.
        try { return !!L.source; } catch (e) { return false; }
    }

    function buildOverlay(comp, tmpl, ov, folder) {
        if (!ov || !ov.asset) { return; }
        var emote = importSource(ov.asset);      // png/gif punch-in image
        if (folder) { emote.parentFolder = folder; }
        tmpl.copyToComp(comp);                   // clone the animated placeholder -> layer(1)
        var clone = comp.layer(1), L;
        if (hasSource(clone)) {
            clone.replaceSource(emote, false);   // swap the emote in, keep the keyframes
            L = clone;
        } else {
            // The "Overlay Style" template is a shape/text/null layer -- nothing to
            // replace. Add the emote and move the template's animation onto it.
            L = comp.layers.add(emote);
            copyTransform(clone, L);
            clone.remove();
        }
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

    // AE clamps EVERY time value -- a layer's startTime, a comp's duration, a Time Remap
    // value -- to +/-10800s (3h), so a layer cannot reach past the 3-hour mark of a source
    // at all. `ae_export.split_source` therefore ships a 6h+ VOD as 2.5h parts and tags
    // each clip with the part holding it; `sources[i].offset` is that part's source time.
    // Clip times stay ABSOLUTE in the manifest -- subtract the offset here and only here.
    function partFor(M, clip) {
        var srcs = M.sources;
        if (!srcs || !srcs.length) { return { path: M.source, offset: 0 }; }
        return srcs[Math.min(clip.src || 0, srcs.length - 1)];
    }

    // Dead-air removal: the manifest's per-clip `keep` lists the source spans worth
    // playing (clip-relative). Time Remap plays them back-to-back -- a linear key pair
    // per span, the second landing a frame early so the next span's key reads as a hard
    // jump cut instead of a ramp. Captions/zoom/overlays are already on this tight
    // timeline (ae_export retimes them). Returns false when the clip has no `keep`.
    function applyKeep(foot, clip, part, dur, fps) {
        var keep = clip.keep;
        if (!keep || !keep.length) { return false; }
        var base = clip.start - (part.offset || 0);   // source time of the clip head
        // startTime 0 so layer time == comp time and the remap keys land where we say
        foot.startTime = 0;
        foot.timeRemapEnabled = true;
        var tr = foot.property("Time Remap"), j, k;
        // Enabling Time Remap stamps two keys (layer in/out). Thin them to at most the
        // one at t=0 -- our own first key overwrites it -- rather than emptying the
        // property, which AE refuses on some layers.
        for (j = tr.numKeys; j >= 2; j--) { tr.removeKey(j); }
        if (tr.numKeys === 1 && tr.keyTime(1) > 1e-6) { try { tr.removeKey(1); } catch (e) {} }
        var f = 1 / fps, t = 0, times = [], vals = [];
        for (k = 0; k < keep.length; k++) {
            var len = keep[k][1] - keep[k][0];
            if (len <= 2 * f) { continue; }            // too short to survive the cut
            times.push(t);              vals.push(base + keep[k][0]);
            times.push(t + len - f);    vals.push(base + keep[k][1] - f);
            t += len;
        }
        if (times.length < 2) { foot.timeRemapEnabled = false; return false; }
        tr.setValuesAtTimes(times, vals);
        for (k = 1; k <= times.length; k++) {
            try {
                tr.setInterpolationTypeAtKey(k, KeyframeInterpolationType.LINEAR,
                                                KeyframeInterpolationType.LINEAR);
            } catch (e) {}
        }
        foot.inPoint = 0;
        foot.outPoint = dur;
        return true;
    }

    function buildClip(M, clip, srcs, tmpl, zoomScale, idx, folder, overlayTmpl, ovFolder,
                       capOff) {
        // `dur` is the tightened length when dead air was cut, else the raw clip span
        var dur = clip.dur || (clip.end - clip.start);
        var comp = app.project.items.addComp(
            "reFire clip " + idx, M.out_w, M.out_h, 1, dur, M.fps);
        if (folder) { comp.parentFolder = folder; }

        var src = srcs[0];
        var part = partFor(M, clip);
        var foot = comp.layers.add(importSource(part.path));   // cached by path
        foot.name = FOOT_NAME;                  // how `update` finds it again
        if (!applyKeep(foot, clip, part, dur, M.fps)) {
            foot.startTime = -(clip.start - (part.offset || 0));
            foot.inPoint = 0;
            foot.outPoint = dur;
        }

        var tr = foot.property("Transform");
        tr.property("Anchor Point").setValue([0, src.height]);
        tr.property("Position").setValue([0, M.out_h]);
        applyZoom(zoomScale, tr.property("Scale"), dur,
                  M.out_w / src.width * 100, clip.zoom_episodes);

        buildCaptions(M, comp, tmpl, clip.captions, capOff, folder);
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
            // style pass: a hook section sets card:false so the cut starts hot (no title card)
            if (sections[s].title && sections[s].card !== false) {
                seq.push(buildCard(M, sections[s].title, cardTmpl, folder));
            }
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

    function doBuild(manifestPath, capOff) {
        var M = readManifest(manifestPath);
        if (!M) { return "Manifest not found:\n" + manifestPath; }
        if (!M.clips || !M.clips.length) { return "Manifest has no clips to build."; }
        app.beginUndoGroup("reFire build");
        try {
            var tplF = ensureFolder("reFire templates");   // shared singletons
            var footF = ensureFolder("reFire footage");
            var buildF = nextBuildFolder();                // this build's home

            var ovF = ensureFolder("reFire overlays");      // shared emote/sfx/music assets
            // one item per source part (just the VOD itself when it's under 2.5h)
            var srcs = [], paths = [], si;
            if (M.sources && M.sources.length) {
                for (si = 0; si < M.sources.length; si++) { paths.push(M.sources[si].path); }
            } else {
                paths.push(M.source);
            }
            for (si = 0; si < paths.length; si++) {
                var it = importSource(paths[si]);
                it.parentFolder = footF;
                srcs.push(it);
            }
            var tmpl = ensureTemplate(M);
            var zoomScale = ensureZoomTemplate(M);
            var cardTmpl = ensureSectionTemplate(M);
            var overlayTmpl = ensureOverlayTemplate(M);
            // tuck the (shared) templates into their folder
            var capC = findComp("Caption Template"); if (capC) { capC.parentFolder = tplF; }
            var zoomC = findComp("Zoom Template");   if (zoomC) { zoomC.parentFolder = tplF; }
            var ovC = findComp("Overlay Template");  if (ovC) { ovC.parentFolder = tplF; }
            if (cardTmpl) { cardTmpl.parentFolder = tplF; }

            var comps = [];
            for (var c = 0; c < M.clips.length; c++) {
                comps.push(buildClip(M, M.clips[c], srcs, tmpl, zoomScale, c, buildF,
                                     overlayTmpl, ovF, capOff));
            }
            buildMaster(M, comps, cardTmpl, buildF);
            return "Built " + comps.length + " clip(s) in '" + buildF.name +
                   "'. Master in render queue.";
        } finally {
            app.endUndoGroup();
        }
    }

    // --- update: refresh captions + zoom, keep footage ---------------------

    function doUpdate(manifestPath, capOff) {
        var M = readManifest(manifestPath);
        if (!M) { return "Manifest not found:\n" + manifestPath; }
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
                for (var i = comp.numLayers; i >= 1; i--) {   // drop old overlays
                    var nm = comp.layer(i).name;
                    if (nm === "reFire overlay" || nm === "reFire sfx") {
                        comp.layer(i).remove();
                    }
                }
                dropCaptions(comp);
                var foot = footageLayer(comp);   // re-punch zoom from the template
                if (foot) {
                    applyZoom(zoomScale, foot.property("Transform").property("Scale"),
                              comp.duration, M.out_w / foot.source.width * 100,
                              M.clips[c].zoom_episodes);
                }
                buildCaptions(M, comp, tmpl, M.clips[c].captions, capOff,
                              comp.parentFolder);
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

    // --- shift: re-stamp captions only, at a new offset ---------------------

    // Word timestamps drift a little against the audio, so the caption timing in the
    // manifest can read early or late. This slides each clip's "cc" caption comp to
    // `capOff` seconds (+late / -early), leaving footage, zoom and overlays alone --
    // so it is fast enough to nudge, look, nudge again. The offset is absolute: it is
    // the layer's startTime, so it never accumulates.
    function doShift(manifestPath, capOff) {
        var M = readManifest(manifestPath);
        if (!M) { return "Manifest not found:\n" + manifestPath; }
        if (!findComp("Caption Template")) { return "No template yet -- run Build first."; }
        var tmpl = ensureTemplate(M);
        var off = Number(capOff) || 0;
        app.beginUndoGroup("reFire caption shift");
        try {
            var touched = 0;
            for (var c = 0; c < M.clips.length; c++) {
                var comp = findComp("reFire clip " + c);
                if (!comp) { continue; }
                var L = captionLayer(comp);
                if (L) { shiftCaptions(L, off); }
                else {                       // built before captions were precomped
                    dropCaptions(comp);
                    buildCaptions(M, comp, tmpl, M.clips[c].captions, off,
                                  comp.parentFolder);
                }
                touched++;
            }
            if (!touched) { return "No reFire clip comps found -- run Build first."; }
            return "Captions shifted " + (off >= 0 ? "+" : "") + off.toFixed(2) +
                   "s across " + touched + " clip(s).";
        } finally {
            app.endUndoGroup();
        }
    }

    // --- public API --------------------------------------------------------

    // CEP collapses ANY uncaught ExtendScript exception into the opaque string
    // "EvalScript error." (or an empty result), so a real failure -- a missing
    // source file, a locked comp, a bad template -- would surface in the panel as
    // nothing useful. Catch at the boundary and hand back the actual message and
    // line instead. Everything below always resolves to a printable string.
    function guard(fn) {
        return function (path, arg) {
            try {
                var r = fn(path, arg);
                return (r === undefined || r === null) ? "Done." : String(r);
            } catch (e) {
                var msg = "Error: " + (e.message || e.toString());
                if (e.line) { msg += "  [reFire.jsx:" + e.line + "]"; }
                return msg;
            }
        };
    }

    // the panel probes this on load to prove the library reached AE's engine
    function probe() {
        return "reFire.jsx ok - AE " + app.version + ", project '" +
               (app.project.file ? decodeURI(app.project.file.name) : "untitled") + "'";
    }

    return { build: guard(doBuild), update: guard(doUpdate), shift: guard(doShift),
             probe: guard(probe) };
}());
