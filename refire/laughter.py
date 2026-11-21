"""Sustained laughter from YAMNet: Google's official weights, a numpy forward pass.

No TensorFlow: the frontend (log-mel patches) and the MobileNet-style net are re-implemented
from models/research/audioset/yamnet (features.py, yamnet.py, params.py) and run on the
published yamnet.h5. Used by punch (P2: laughter counts only when sustained >= 1.0 s).

    laugh_segments(wav, [(a, b), ...]) -> [(src_t0, src_t1, prob)]   # transcript clock
"""
from __future__ import annotations

import csv
import hashlib
import urllib.request
import wave
from pathlib import Path

import numpy as np

MODELS = Path.home() / ".cache" / "refire" / "models"
WEIGHTS = ("https://storage.googleapis.com/audioset/yamnet.h5", "yamnet.h5",
           "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744")
CLASSES = ("https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/"
           "yamnet_class_map.csv", "yamnet_class_map.csv",
           "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2")
LAUGH = ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle")   # not baby laughter

SR = 16000
STFT_WIN, STFT_HOP, FFT = 400, 160, 512          # 25 ms / 10 ms, fft = next pow2
MEL_BINS, MEL_LO, MEL_HI = 64, 125.0, 7500.0
PATCH, PATCH_HOP = 96, 48                        # frames: 0.96 s window, 0.48 s hop
PATCH_S, HOP_S = 0.96, 0.48
LOG_OFFSET, BN_EPS = 0.001, 1e-4
LAYERS = [(2, 32)] + [(s, c) for s, c in ((1, 64), (2, 128), (1, 128), (2, 256), (1, 256), (2, 512),
                                         (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (2, 1024),
                                         (1, 1024))]
THRESHOLD = 0.3   # ponytail: laugh score cut, tune on real streams (speech over laughter scores low)


def _fetch(spec) -> Path:
    url, name, sha = spec
    p = MODELS / name
    if not p.exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "refire"}),
                                    timeout=120) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError(f"checksum mismatch for {url}")
        p.write_bytes(data)
    return p


# --------------------------------------------------------------------------- frontend

def _mel_matrix() -> np.ndarray:
    """tf.signal.linear_to_mel_weight_matrix (htk mel, dc bin zeroed)."""
    def mel(f):
        return 1127.0 * np.log1p(np.asarray(f, np.float64) / 700.0)
    bins = FFT // 2 + 1
    spec_mel = mel(np.linspace(0, SR / 2, bins)[1:])[:, None]
    edges = np.linspace(mel(MEL_LO), mel(MEL_HI), MEL_BINS + 2)
    lo, ce, hi = edges[:-2], edges[1:-1], edges[2:]
    w = np.maximum(0.0, np.minimum((spec_mel - lo) / (ce - lo), (hi - spec_mel) / (hi - ce)))
    return np.vstack([np.zeros((1, MEL_BINS)), w]).astype(np.float32)


_MEL = _mel_matrix()
_HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(STFT_WIN) / STFT_WIN)).astype(np.float32)   # periodic


def patches(wav: np.ndarray) -> np.ndarray:
    """float waveform in [-1, 1] at 16 kHz -> [n, 96, 64] log-mel patches (yamnet's padding)."""
    min_n = int(round((PATCH_S + (STFT_WIN - STFT_HOP) / SR) * SR))       # 15600
    hop_n = int(HOP_S * SR)
    n = max(len(wav), min_n)
    extra = n - min_n
    pad = (min_n - len(wav) if len(wav) < min_n else 0) + (-extra) % hop_n
    w = np.pad(wav.astype(np.float32), (0, max(0, pad)))
    nf = 1 + (len(w) - STFT_WIN) // STFT_HOP
    idx = np.arange(STFT_WIN)[None, :] + STFT_HOP * np.arange(nf)[:, None]
    mag = np.abs(np.fft.rfft(w[idx] * _HANN, n=FFT)).astype(np.float32)
    logmel = np.log(mag @ _MEL + LOG_OFFSET)
    npatch = 1 + (nf - PATCH) // PATCH_HOP
    pidx = np.arange(PATCH)[None, :] + PATCH_HOP * np.arange(npatch)[:, None]
    return logmel[pidx]


# --------------------------------------------------------------------------- net

class YAMNet:
    def __init__(self):
        import h5py
        f = h5py.File(_fetch(WEIGHTS), "r")

        def g(path):
            name = path.rsplit("/", 1)
            return np.asarray(f[f"{name[0]}/{path}"], np.float32)

        def bn(prefix):
            return (g(f"{prefix}/bn/moving_mean:0"), g(f"{prefix}/bn/moving_variance:0"), g(f"{prefix}/bn/beta:0"))

        def fold(kernel, prefix):
            """batchnorm (no scale term) folded into the conv: y = x*k*s + (beta - mean*s)."""
            mean, var, beta = bn(prefix)
            sc = 1.0 / np.sqrt(var + BN_EPS)
            return (kernel * sc).astype(np.float32), (beta - mean * sc).astype(np.float32)

        self.l1 = fold(g("layer1/conv/kernel:0")[:, :, 0, :], "layer1/conv")              # [3, 3, 32]
        self.sep = [fold(g(f"layer{i}/depthwise_conv/depthwise_kernel:0")[:, :, :, 0], f"layer{i}/depthwise_conv")
                    + fold(g(f"layer{i}/pointwise_conv/kernel:0")[0, 0], f"layer{i}/pointwise_conv")
                    for i in range(2, 15)]
        self.wl, self.bl = g("logits/kernel:0"), g("logits/bias:0")
        f.close()
        with open(_fetch(CLASSES), encoding="utf-8") as fh:
            self.names = [r["display_name"] for r in csv.DictReader(fh)]

    @staticmethod
    def _same_pad(x, stride):
        """tf 'same' padding for a 3x3 kernel: extra pixel goes bottom/right."""
        pads = []
        for n in x.shape[1:3]:
            total = max((-(-n // stride) - 1) * stride + 3 - n, 0)
            pads.append((total // 2, total - total // 2))
        return np.pad(x, ((0, 0), pads[0], pads[1], (0, 0)))

    def _depthwise(self, x, k, bias, stride):
        """3x3 per-channel conv (tf 'same' padding) + folded bn + relu; k [3, 3, C]."""
        xp = self._same_pad(x, stride)
        ho, wo = -(-x.shape[1] // stride), -(-x.shape[2] // stride)
        acc = np.zeros((x.shape[0], ho, wo, k.shape[-1]), np.float32)
        for i in range(3):
            for j in range(3):
                acc += xp[:, i:i + stride * ho:stride, j:j + stride * wo:stride, :] * k[i, j]
        acc += bias
        return np.maximum(acc, 0.0, out=acc)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        """[n, 96, 64] patches -> [n, 521] class scores (sigmoid)."""
        x = self._depthwise(p[..., None].astype(np.float32), self.l1[0], self.l1[1], 2)  # 1 -> 32 channels
        for (stride, _), (dk, db, pk, pb) in zip(LAYERS[1:], self.sep):
            x = self._depthwise(x, dk, db, stride)
            n, h, w, c = x.shape
            x = x.reshape(-1, c) @ pk + pb             # one gemm, not n*h tiny ones
            x = np.maximum(x, 0.0, out=x).reshape(n, h, w, -1)
        logits = x.mean(axis=(1, 2)) @ self.wl + self.bl
        return 0.5 * (1.0 + np.tanh(0.5 * logits))   # sigmoid without exp overflow

    def scores(self, wav: np.ndarray, batch: int = 512) -> np.ndarray:   # 512: 6 ms/patch vs 10 at 128
        ps = patches(wav)
        return np.vstack([self(ps[i:i + batch]) for i in range(0, len(ps), batch)]) if len(ps) else \
            np.zeros((0, len(self.wl[0])), np.float32)


# --------------------------------------------------------------------------- run io

def read_wav(path, a: float, b: float) -> np.ndarray:
    """[a, b) seconds of a 16 kHz mono int16 wav as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != SR or wf.getnchannels() != 1:
            raise ValueError(f"{path}: want 16 kHz mono, got {wf.getframerate()} Hz x{wf.getnchannels()}")
        s0 = max(0, int(a * SR))
        wf.setpos(min(s0, wf.getnframes()))
        raw = wf.readframes(max(0, int(b * SR) - s0))
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def _merge(ranges, pad):
    out = []
    for a, b in sorted((max(0.0, a - pad), b + pad) for a, b in ranges):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def segments(times: np.ndarray, laugh: np.ndarray, thr: float = THRESHOLD) -> list[tuple[float, float, float]]:
    """runs of patches scoring >= thr -> (start, end, max score); one patch spans PATCH_S."""
    out, run = [], []
    for t, s in zip(times, laugh):
        if s >= thr:
            run.append((t, s))
            continue
        if run:
            out.append((run[0][0], run[-1][0] + PATCH_S, max(x for _, x in run)))
            run = []
    if run:
        out.append((run[0][0], run[-1][0] + PATCH_S, max(x for _, x in run)))
    return [(round(a, 3), round(b, 3), round(float(p), 3)) for a, b, p in out]


def laugh_segments(wav_path, ranges, net: YAMNet | None = None, thr: float = THRESHOLD,
                   pad: float = 1.0) -> list[tuple[float, float, float]]:
    """laughter runs inside the used `ranges` (seconds on the wav's clock, +-pad)."""
    net = net or YAMNet()
    cols = [net.names.index(n) for n in LAUGH]
    spans = _merge(ranges, pad)
    ps = [patches(read_wav(wav_path, a, b)) for a, b in spans]
    if not ps:
        return []
    allp = np.concatenate(ps)                    # one pass in full batches: a cut is ~200 short pieces
    sc = np.vstack([net(allp[i:i + 512]) for i in range(0, len(allp), 512)])[:, cols].max(axis=1)
    out, k = [], 0
    for (a, _), p in zip(spans, ps):
        out += segments(a + HOP_S * np.arange(len(p)), sc[k:k + len(p)], thr)
        k += len(p)
    return out
