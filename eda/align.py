"""Broadcast -> tactical time alignment by OCR of the broadcast scoreboard clock.

    .venv/bin/python eda/align.py            # ~10 min: one ffmpeg pass (1 crop / 2 s) + tesseract
    from eda.align import b2t; b2t(1200.0)   # broadcast seconds -> tactical frame (or None off-play)

The LaLiga clock runs continuously, so within a half tactical_frame = kickoff + 25*(clock - half_start)
and the only unknown per half is the broadcast->clock offset. Writes eda/out/ocr/clock_ocr.csv (all
reads) and eda/align.json (two offsets + residual stats; derived constants, fine to commit).
"""
import csv, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from games import asset_paths

OUT = os.path.join(ROOT, "eda/out/ocr"); JSON = os.path.join(ROOT, "eda/align.json")
STEP = 2.0                                   # seconds between samples
BOX = (95, 122, 90, 46)                      # x, y, w, h of the clock in the 1920x1080 frame
KICKOFF = {1: 11185, 2: 113742}; HALF_START = {1: 0, 2: 2700}   # tactical frames / clock seconds

def ocr(png):
    from PIL import Image, ImageOps
    im = ImageOps.invert(Image.open(png).convert("L")); im = im.resize((im.width * 4, im.height * 4), Image.LANCZOS).point(lambda v: 255 if v > 110 else 0)
    im.save(png)
    r = subprocess.run(["tesseract", png, "-", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789:"], capture_output=True, text=True)
    m = re.fullmatch(r"(\d{1,3}):(\d\d)", r.stdout.strip()); return int(m[1]) * 60 + int(m[2]) if m and int(m[2]) < 60 else None

def run():
    os.makedirs(OUT, exist_ok=True)
    x, y, w, h = BOX
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", asset_paths(source="broadcast")["video"], "-vf", f"fps=1/{STEP},crop={w}:{h}:{x}:{y}", os.path.join(OUT, "c_%05d.png")], check=True)
    pngs = sorted(f for f in os.listdir(OUT) if f.startswith("c_"))
    with ThreadPoolExecutor(8) as ex: reads = list(ex.map(ocr, [os.path.join(OUT, f) for f in pngs]))
    rows = [(i * STEP, c) for i, c in enumerate(reads)]
    with open(os.path.join(OUT, "clock_ocr.csv"), "w", newline="") as fh: csv.writer(fh).writerows([("broadcast_sec", "clock_sec")] + rows)
    fit(rows)

def fit(rows):
    bs = np.array([b for b, c in rows if c is not None]); cs = np.array([c for b, c in rows if c is not None])
    half = np.where(cs < 2700, 1, 2); res = {}
    for hh in (1, 2):
        m = half == hh; off = cs[m] - bs[m]; med = float(np.median(off)); ok = np.abs(off - med) <= 1.0   # ponytail: median offset; OCR misreads are sparse outliers
        # clock shows floor(seconds) and samples sit on whole seconds -> true offset is in [med, med+1): use the midpoint
        res[hh] = {"offset": med + 0.5, "n": int(m.sum()), "inliers": int(ok.sum()), "resid_p95": float(np.percentile(np.abs(off[ok] - med), 95)),
                   "broadcast_range": [float(bs[m][ok].min()), float(bs[m][ok].max())], "clock_range": [int(cs[m][ok].min()), int(cs[m][ok].max())]}
        print(f"half {hh}: offset {med:.1f}s  reads {m.sum()}  inliers {ok.sum()}  |resid| p95 {res[hh]['resid_p95']:.2f}s  clock {cs[m].min()}-{cs[m].max()}s")
    json.dump(res, open(JSON, "w"), indent=1); print("wrote", JSON)

def b2t(bsec, _p=[]):
    """Broadcast seconds -> tactical frame. None outside the halves. Clock = bsec + offset."""
    if not _p: _p.append({int(k): v for k, v in json.load(open(JSON)).items()})
    for hh, p in _p[0].items():
        lo, hi = p["broadcast_range"]; lo = min(lo, HALF_START[hh] - p["offset"])   # half starts at kickoff, not at the first clock read
        if lo <= bsec <= hi + 5: return int(round(KICKOFF[hh] + 25 * (bsec + p["offset"] - HALF_START[hh])))
    return None

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--refit":
        fit([(float(b), int(c) if c else None) for b, c in csv.reader(open(os.path.join(OUT, "clock_ocr.csv"))) if b != "broadcast_sec"])
    else: run()
