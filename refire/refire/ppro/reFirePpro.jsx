// reFire — Premiere Pro automation (ExtendScript, ES3).
//
// This file is the Premiere half of the reFire CEP panel; the UI lives next door in
// index.html and calls in here via evalScript. Install the panel with
// `refire/ppro/install.ps1`, then Window > Extensions > reFire.
//
// SCOPE: clip selection + subtitling, nothing else. Premiere's ExtendScript cannot
// create text layers, cannot set keyframe easing and has no documented transition
// API, so the zoom punches, emote overlays, SFX, section cards and music bed that
// the After Effects build (`refire/ae/reFire.jsx`) produces are deliberately NOT
// built here -- you add those by hand. What this gives you is the tedious part:
// every chosen clip, in story order, dead air already cut, with a matching .srt.
//
// API (everything else in here is private):
//   reFirePpro.build(manifestPath)   import the clip proxies + lay them out on V1 of
//                                    a fresh sequence, one trackItem per kept span.
//   reFirePpro.timeline(manifestPath) dump the ACTIVE sequence's V1 to timeline.tsv, so
//                                    `refire recap` can re-caption a hand re-cut edit.
//   reFirePpro.attach(manifestPath, srtPath)  import a .srt into the reFire bin.
//   reFirePpro.probe()               liveness check the panel runs on load.
// All return a human-readable status string, which the panel shows and logs.
//
// CAPTIONS: the panel runs `python -m refire srt <manifest> --offset X` before it
// calls build, which writes captions.srt next to the manifest in MASTER-TIMELINE
// time (see refire/srt.py -- it walks the clips with the same layout rule this file
// does, so the two line up by construction). Nudging captions = change the offset
// field and Build again; the srt regen is instant, unlike re-running `make`.
//
// RECAPTION: once you re-cut that sequence by hand the manifest no longer describes
// it, so captions.srt drifts. The panel's Recaption button dumps the timeline through
// `timeline()` and runs `python -m refire recap <manifest>`, which re-groups the run's
// transcript against the source ranges the timeline is ACTUALLY showing -- moves,
// trims, splits and handles pulled wider than reFire's own cut all come out right.

var reFirePpro = (function () {

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

    // `ae_export` writes one all-intra proxy per clip (or, on the raw-VOD path, 2.5h
    // parts) into `sources`, and tags each clip with its index. Clip times stay
    // ABSOLUTE in the manifest -- subtract `offset` here and only here.
    function partFor(M, clip) {
        var srcs = M.sources;
        if (!srcs || !srcs.length) { return { path: M.source, offset: 0 }; }
        return srcs[Math.min(clip.src || 0, srcs.length - 1)];
    }

    // Clips in the order the sequence lays them out. Mirrors refire/srt.py:_ordered_clips
    // -- if these two ever disagree, every caption after the first drifts.
    function orderedClips(M) {
        var clips = M.clips || [], out = [], s, i, idxs, k;
        var sections = M.sections || [];
        if (!sections.length) { return clips; }
        for (s = 0; s < sections.length; s++) {
            idxs = sections[s].clip_indices || [];
            for (k = 0; k < idxs.length; k++) {
                i = idxs[k];
                if (i >= 0 && i < clips.length) { out.push(clips[i]); }
            }
        }
        return out.length ? out : clips;
    }

    // --- project helpers ---------------------------------------------------

    function seconds(t) {
        var T = new Time();
        T.seconds = t;
        return T;
    }

    function normPath(p) {
        return String(p).replace(/\\/g, "/").toLowerCase();
    }

    function ensureBin(name) {
        // reuse the bin across builds so repeat Builds don't clutter the project
        var root = app.project.rootItem, i;
        for (i = 0; i < root.children.numItems; i++) {
            var it = root.children[i];
            if (it.name === name && it.type === ProjectItemType.BIN) { return it; }
        }
        return root.createBin(name);
    }

    // Import every path, then hand back the ProjectItems in the SAME order. Import is
    // by media path, so an already-imported proxy is matched instead of duplicated.
    function importSources(paths, targetBin) {
        var want = {}, missing = [], i;
        for (i = 0; i < paths.length; i++) {
            var f = new File(paths[i]);
            if (!f.exists) { missing.push(paths[i]); }
            want[normPath(f.fsName)] = true;
        }
        if (missing.length) {
            throw new Error("Source video not found:\n" + missing.join("\n"));
        }
        // importFiles skips paths already in the project, so this is safe to re-run
        app.project.importFiles(paths, true, targetBin, false);

        // index everything in the project by media path -- an earlier build may have
        // put the item somewhere other than our bin.
        var byPath = {};
        indexItems(app.project.rootItem, byPath);
        var items = [];
        for (i = 0; i < paths.length; i++) {
            var key = normPath(new File(paths[i]).fsName);
            if (!byPath[key]) { throw new Error("Import failed for:\n" + paths[i]); }
            items.push(byPath[key]);
        }
        return items;
    }

    function indexItems(node, byPath) {
        for (var i = 0; i < node.children.numItems; i++) {
            var it = node.children[i];
            if (it.type === ProjectItemType.BIN) {
                indexItems(it, byPath);
            } else {
                var mp = "";
                try { mp = it.getMediaPath(); } catch (e) {}
                if (mp) { byPath[normPath(mp)] = it; }
            }
        }
    }

    function nextSequenceName() {
        // a fresh "reFire cut 1" / "reFire cut 2" / ... so repeated Builds don't mingle
        var n = 1;
        for (;;) {
            var nm = "reFire cut " + n, taken = false;
            for (var i = 0; i < app.project.sequences.numSequences; i++) {
                if (app.project.sequences[i].name === nm) { taken = true; break; }
            }
            if (!taken) { return nm; }
            n++;
        }
    }

    function newSequence(name, firstItem, targetBin) {
        // From a clip, so the sequence settings match the media and nothing has to be
        // guessed. Older builds return nothing and just make it active; newer ones
        // return the Sequence.
        var seq = null;
        try {
            seq = app.project.createNewSequenceFromClips(name, [firstItem], targetBin);
        } catch (e) { seq = null; }
        if (!seq) { seq = app.project.activeSequence; }
        if (!seq || seq.name !== name) {
            // last resort: a default-preset sequence. The user fixes the settings once.
            app.project.createNewSequence(name, "refire-" + (new Date()).getTime());
            seq = app.project.activeSequence;
        }
        if (!seq) { throw new Error("Could not create a sequence."); }
        clearTracks(seq);          // drop whatever createNewSequenceFromClips laid down
        return seq;
    }

    function clearTracks(seq) {
        var groups = [seq.videoTracks, seq.audioTracks], g, t, c;
        for (g = 0; g < groups.length; g++) {
            for (t = 0; t < groups[g].numTracks; t++) {
                var track = groups[g][t];
                for (c = track.clips.numItems - 1; c >= 0; c--) {
                    try { track.clips[c].remove(false, false); } catch (e) {}
                }
            }
        }
    }

    // --- build -------------------------------------------------------------

    function doBuild(manifestPath) {
        var M = readManifest(manifestPath);
        if (!M) { return "Manifest not found:\n" + manifestPath; }
        if (!M.clips || !M.clips.length) { return "Manifest has no clips to build."; }

        var bin = ensureBin("reFire");
        var paths = [], i;
        if (M.sources && M.sources.length) {
            for (i = 0; i < M.sources.length; i++) { paths.push(M.sources[i].path); }
        } else {
            paths.push(M.source);
        }
        var items = importSources(paths, bin);

        var clips = orderedClips(M);
        var seq = newSequence(nextSequenceName(), items[0], bin);
        var v0 = seq.videoTracks[0];

        // The playhead is the PLACED end of the previous cut, not a running sum of the
        // requested durations. Premiere snaps in/out points and insert times to whole
        // frames, so summing raw float spans drifts a fraction of a frame per cut and
        // sooner or later lands the next clip 1-2 frames past the last one -- which
        // reads as black flashes between clips. Reading the real end back is exact and
        // needs no fps.
        var at = seconds(0), playhead = 0.0, spansPlaced = 0, c, s;
        for (c = 0; c < clips.length; c++) {
            var clip = clips[c];
            var part = partFor(M, clip);
            var item = items[Math.min(clip.src || 0, items.length - 1)];
            var base = clip.start - (part.offset || 0);      // source time of the clip head
            // `keep` lists the clip-relative source spans worth playing (dead air already
            // dropped upstream). No `keep` -> the whole clip is one span. Each span becomes
            // its own trackItem, so the jump cuts are real cuts -- no Time Remap needed,
            // which is the one place Premiere is easier than AE here.
            var spans = clip.keep;
            if (!spans || !spans.length) { spans = [[0, clip.end - clip.start]]; }
            for (s = 0; s < spans.length; s++) {
                var a = base + spans[s][0], b = base + spans[s][1];
                if (b - a <= 1e-3) { continue; }
                // mediaType is set per-stream rather than with a combined constant --
                // the "both" value differs across Premiere versions, 1 and 2 do not.
                item.setInPoint(a, 1);  item.setInPoint(a, 2);
                item.setOutPoint(b, 1); item.setOutPoint(b, 2);
                v0.overwriteClip(item, at);
                var placed = v0.clips.numItems ? v0.clips[v0.clips.numItems - 1] : null;
                // if the overwrite silently no-op'd, leaving `at` put makes the next
                // clip land here instead of leaving a hole
                if (placed) { at = placed.end; playhead = at.seconds; }
                spansPlaced++;
            }
        }

        var msg = "Built '" + seq.name + "': " + clips.length + " clip(s) in " +
                  spansPlaced + " cut(s), " + playhead.toFixed(1) + "s.";
        if (seq.audioTracks.numTracks && seq.audioTracks[0].clips.numItems === 0) {
            msg += "  NOTE: no audio landed on A1 -- linked audio did not follow.";
        }
        msg += "  " + attachCaptions(manifestPath, bin);
        return msg;
    }

    // Premiere has no documented way to put a caption track on a sequence from
    // ExtendScript, so the .srt is imported into the bin and dragged in by hand --
    // one drag. If even the import is refused, hand back the path for File > Import.
    function attachCaptions(manifestPath, bin) {
        var srt = new File(new File(manifestPath).parent.fsName + "/captions.srt");
        if (!srt.exists) {
            return "No captions.srt beside the manifest (run the panel's Build, "
                 + "which regenerates it).";
        }
        try {
            app.project.importFiles([srt.fsName], true, bin, false);
            return "captions.srt is in the reFire bin -- drag it onto the sequence.";
        } catch (e) {
            return "Captions: File > Import  " + srt.fsName;
        }
    }

    // --- recut -------------------------------------------------------------

    // Dump the ACTIVE sequence's V1 to timeline.tsv beside the manifest -- one row per
    // trackItem: media path, source in, source out, timeline in, timeline out. That is
    // everything `refire recap` needs to put captions back under footage a human has
    // since re-cut: source in/out plus the manifest's sources[].offset gives the absolute
    // VOD range the clip is showing, and timeline in/out gives where it now plays (and,
    // by ratio, whether it was retimed).
    //
    // TSV, not JSON: ExtendScript is ES3 and has no JSON.stringify, and no media path
    // contains a tab or a newline, so the format cannot go ambiguous on us.
    //
    // ponytail: V1 only -- that is where doBuild lays the spine down, and anything
    // stacked above it reads as b-roll nobody wants captioned. Loop seq.videoTracks
    // here if a stacked recut ever becomes the normal way to work.
    function doTimeline(manifestPath) {
        var seq = app.project.activeSequence;
        if (!seq) { return "Error: no active sequence -- open your recut sequence first."; }
        if (!seq.videoTracks.numTracks) {
            return "Error: '" + seq.name + "' has no video tracks.";
        }
        var track = seq.videoTracks[0], rows = [], i;
        for (i = 0; i < track.clips.numItems; i++) {
            var c = track.clips[i], mp = "";
            if (c.disabled) { continue; }      // a muted clip does not play, so no captions
            try { mp = c.projectItem.getMediaPath(); } catch (e) {}
            if (!mp) { continue; }             // titles, colour mattes, offline media
            // toFixed keeps long VOD timecodes out of exponent notation, which the
            // python side would read as a bad row and silently drop.
            rows.push([mp,
                       c.inPoint.seconds.toFixed(6), c.outPoint.seconds.toFixed(6),
                       c.start.seconds.toFixed(6), c.end.seconds.toFixed(6)].join("\t"));
        }
        if (!rows.length) {
            return "Error: nothing on V1 of '" + seq.name + "' to caption.";
        }

        var out = new File(new File(manifestPath).parent.fsName + "/timeline.tsv");
        out.encoding = "UTF-8";
        if (!out.open("w")) { throw new Error("Cannot write " + out.fsName); }
        out.write(rows.join("\n"));
        out.close();
        return "Timeline: " + rows.length + " clip(s) on V1 of '" + seq.name + "'.";
    }

    // Same one-drag hand-off as attachCaptions, for the .srt `recap` just wrote. It is a
    // fresh captions.recutN.srt every press by design: importFiles() skips a path already
    // in the project, so reusing one name would hand back the caption track being replaced.
    function doAttach(manifestPath, srtPath) {
        var srt = new File(srtPath);
        if (!srt.exists) { return "Error: captions not found: " + srtPath; }
        try {
            app.project.importFiles([srt.fsName], true, ensureBin("reFire"), false);
            return "Recaptioned: " + srt.name + " is in the reFire bin. Delete the old "
                 + "caption track, then drag this one onto the sequence.";
        } catch (e) {
            return "Recaptioned. File > Import  " + srt.fsName;
        }
    }

    // --- public API --------------------------------------------------------

    // CEP collapses ANY uncaught ExtendScript exception into the opaque string
    // "EvalScript error." (or an empty result), so a real failure -- a missing
    // source file, a locked sequence -- would surface in the panel as nothing
    // useful. Catch at the boundary and hand back the actual message and line.
    function guard(fn) {
        return function (path, arg) {
            try {
                var r = fn(path, arg);
                return (r === undefined || r === null) ? "Done." : String(r);
            } catch (e) {
                var msg = "Error: " + (e.message || e.toString());
                if (e.line) { msg += "  [reFirePpro.jsx:" + e.line + "]"; }
                return msg;
            }
        };
    }

    // the panel probes this on load to prove the library reached Premiere's engine
    function probe() {
        return "reFirePpro.jsx ok - Premiere " + app.version + ", project '" +
               (app.project.name || "untitled") + "'";
    }

    return { build: guard(doBuild), timeline: guard(doTimeline),
             attach: guard(doAttach), probe: guard(probe) };
}());
