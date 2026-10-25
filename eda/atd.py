"""Raw ATD -> numpy, cached. Fast path for the whole team: no kloppy, ~10 s once, instant after.

    from eda.atd import load
    d = load()            # dict: players (F,46,3) x,y,speed in metres ; ball (F,3) ; team (46,) 0=H 1=A
Row i is video frame i (ATD frame_id - 1). NaN = not tracked.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, GAME, require, source_frames
from starter.baseline import team_of_each_track

def load(source="tactical"):
    cache = os.path.join(ROOT, "eda", f"atd_{source}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    meta, raw = require(["atd_meta", "atd_raw"], source=source)
    team = team_of_each_track(meta)
    L, W = GAMES[GAME]["pitch"]
    n = source_frames(source)
    players = np.full((n, len(team), 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    scale = np.array([L, W, 1.0], dtype=np.float32)
    with open(raw) as fh:
        for line in fh:
            fid, ps, b = line.rstrip("\n").split(":")
            i = int(fid) - 1
            players[i] = np.array([g.split(",") for g in ps.split(";")], dtype=np.float32) * scale
            ball[i] = np.array(b.split(","), dtype=np.float32) * scale
    out = {"players": players, "ball": ball, "team": team}
    np.savez_compressed(cache, **out)
    return out

if __name__ == "__main__":
    d = load()
    F = len(d["ball"])
    assert F == 199319 and d["players"].shape[1] == len(d["team"])
    graded = np.zeros(F, bool); graded[11185:90652] = True; graded[110808:190198] = True
    print("ball visible: all %.3f  graded %.3f" % (np.isfinite(d["ball"][:, 0]).mean(),
                                                   np.isfinite(d["ball"][graded, 0]).mean()))
    print("tracked players per graded frame: mean %.1f" % np.isfinite(d["players"][graded, :, 0]).sum(1).mean())
    print("OK")
