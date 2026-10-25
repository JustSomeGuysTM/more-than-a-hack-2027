"""Plot tracking frames on a pitch, with the public GT label where it exists.

    .venv/bin/python eda/plot.py 11185 12700 15889      # one PNG per frame -> eda/out/
    .venv/bin/python eda/plot.py --clip 11185 11600     # a 16 s GIF, every 5th frame
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from starter.utils import draw_pitch, HOME_COLOUR, AWAY_COLOUR
from eda.atd import load

OUT = os.path.join(ROOT, "eda", "out"); os.makedirs(OUT, exist_ok=True)
GT = np.loadtxt(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"),
                dtype=str, delimiter=",", skiprows=1, usecols=1)

def frame(d, i, ax):
    ax.clear(); draw_pitch(ax, 105.3, 68.0)
    p, t, b = d["players"][i], d["team"], d["ball"][i]
    for side, col in ((0, HOME_COLOUR), (1, AWAY_COLOUR)):
        m = (t == side) & np.isfinite(p[:, 0])
        ax.scatter(p[m, 0], p[m, 1], c=col, s=90, edgecolors="k", zorder=3)
    if np.isfinite(b[0]):
        ax.scatter(b[0], b[1], c="w", s=50, edgecolors="k", zorder=4)
    ax.set_title(f"frame {i}  t={i/25:.1f}s  gt={GT[i]}  ball={'yes' if np.isfinite(b[0]) else 'MISSING'}  "
                 f"tracked={np.isfinite(p[:,0]).sum()}")

if __name__ == "__main__":
    d = load()
    args = sys.argv[1:]
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    if args and args[0] == "--clip":
        from matplotlib.animation import FuncAnimation, PillowWriter
        a, b = int(args[1]), int(args[2]); frames = range(a, b, 5)
        anim = FuncAnimation(fig, lambda i: frame(d, i, ax), frames=frames)
        anim.save(os.path.join(OUT, f"clip_{a}_{b}.gif"), writer=PillowWriter(fps=5))
        print("wrote", os.path.join(OUT, f"clip_{a}_{b}.gif"))
    else:
        for i in map(int, args):
            frame(d, i, ax); fig.savefig(os.path.join(OUT, f"frame_{i}.png"), dpi=100)
            print("wrote", os.path.join(OUT, f"frame_{i}.png"))
