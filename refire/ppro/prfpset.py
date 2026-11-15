"""Read Premiere effect presets (.prfpset) so captions follow Nguy's preset files at runtime.

A .prfpset is an XML object graph: elements carry ObjectID, references point at them with
ObjectRef. Walk FilterPresetItem -> FilterPreset -> VideoFilterComponent -> Params.
Param `index` is the preset's `Params Index`, which is also the `properties[j]` index in
Premiere's DOM (spikes S5/S7), so the applier addresses params by it.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

TICKS = 254_016_000_000            # premiere ticks per second
COLOR, CHECKBOX = 5, 4             # ParameterControlType values that need typed decoding
GROUP_TYPES = {11, 12, 16}         # group headers / plugin rows, never applied


@dataclass
class Key:
    t: float                       # seconds from the preset's AnchorInPoint
    value: object
    interp: int                    # 5 = bezier
    tangents: list[str]            # raw remaining fields, undecoded


@dataclass
class Param:
    index: int
    name: str | None
    param_id: int
    control_type: int
    value: object                  # static value (StartKeyframe field 2)
    keys: list[Key] = field(default_factory=list)

    @property
    def skip(self) -> bool:
        """plugin internals (`_ Applied Version`) and group headers are not applied."""
        return self.control_type in GROUP_TYPES or (self.name or "").startswith("_ ")


@dataclass
class Effect:
    match_name: str
    display_name: str
    type: int                      # 0 = scale (keys stretch with the clip)
    anchor_in: float               # seconds
    anchor_out: float
    params: list[Param]

    @property
    def duration(self) -> float:
        return self.anchor_out - self.anchor_in

    def param(self, name: str) -> Param:
        return next(p for p in self.params if p.name == name)


def _value(raw: str, control_type: int):
    if control_type == COLOR:      # 64-bit argb, 16 bits a channel; 8-bit = high byte
        v = int(raw)
        return [(v >> s) >> 8 & 0xFF for s in (48, 32, 16, 0)]
    if raw in ("true", "false"):
        return raw == "true"
    if ":" in raw:                 # point, normalized 0-1
        return [float(x) for x in raw.split(":")]
    return float(raw) if "." in raw or "e" in raw else int(raw)


def _param(index: int, e: ET.Element, anchor_in: int) -> Param:
    ct = int(e.findtext("ParameterControlType") or 0)
    start = (e.findtext("StartKeyframe") or "").split(",")
    keys = []
    if e.findtext("IsTimeVarying") == "true":
        for entry in filter(None, (e.findtext("Keyframes") or "").split(";")):
            f = entry.split(",")
            keys.append(Key((int(f[0]) - anchor_in) / TICKS, _value(f[1], ct), int(f[2]), f[3:]))
    return Param(index, e.findtext("Name"), int(e.findtext("ParameterID") or -1), ct,
                 _value(start[1], ct) if len(start) > 1 else None, keys)


def load(path: str | Path) -> list[Effect]:
    """every effect in the preset, in the preset's (= Effect Controls) order."""
    root = ET.parse(path).getroot()
    objs = {e.get("ObjectID"): e for e in root if e.get("ObjectID")}
    effects = []
    for item in root.iter("FilterPresetItem"):
        for ref in item.find("FilterPresets"):
            fp = objs[ref.get("ObjectRef")]
            comp = objs[fp.find("Component").get("ObjectRef")]
            a_in = int(fp.findtext("AnchorInPoint"))
            params = [_param(int(p.get("Index")), objs[p.get("ObjectRef")], a_in)
                      for p in comp.find("Component/Params")]
            effects.append(Effect(fp.findtext("FilterMatchName"), comp.findtext("Component/DisplayName"),
                                  int(fp.findtext("Type")), a_in / TICKS,
                                  int(fp.findtext("AnchorOutPoint")) / TICKS, params))
    return effects
