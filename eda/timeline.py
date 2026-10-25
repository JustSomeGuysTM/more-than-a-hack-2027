"""Public 5-min window: GT label vs ball visibility vs nearest-player team vs smart-tagging team.
    .venv/bin/python eda/timeline.py   -> eda/out/timeline_public.png
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from eda.atd import load
from starter.load_smart_tagging import load as load_tags

A, B = 11185, 18685
d = load(); P, T, Bl = d["players"][A:B], d["team"], d["ball"][A:B]
gt = np.loadtxt(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"), dtype=str, delimiter=",", skiprows=1, usecols=1)[A:B]
# nearest player team (baseline step 1)
dist = np.linalg.norm(P[:, :, :2] - Bl[:, None, :2], axis=2)          # (F,46), nan where missing
near = np.where(np.isfinite(dist).any(1), T[np.nanargmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)], -1)
near[~np.isfinite(Bl[:, 0])] = -1
# min distance of any player to ball (how "attached" the ball is)
mind = np.min(np.where(np.isfinite(dist), dist, np.inf), axis=1); mind[~np.isfinite(mind)] = np.nan
# smart tagging team per frame
tags = load_tags()
tagteam = np.full(B - A, -1)
for t in tags:
    s, e = t["start_frame"], t["end_frame"]
    team = {"Barcelona": 0, "Atletic Club": 1}.get(t["tags"].get("Team"), -1)
    if team >= 0: tagteam[max(s - A, 0):max(e - A, 0)] = team
code = {"H": 0, "A": 1, "D": 2}
rows = [("ground truth", np.vectorize(code.get)(gt)), ("nearest player", np.where(near < 0, 2, near)),
        ("smart tag Team", np.where(tagteam < 0, 2, tagteam))]
cmap = matplotlib.colors.ListedColormap(["#d62728", "#1f77b4", "#bbbbbb"])
fig, axes = plt.subplots(5, 1, figsize=(16, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 2]})
x = np.arange(A, B) / 25
for ax, (name, arr) in zip(axes, rows):
    ax.imshow(arr[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=2, extent=[x[0], x[-1], 0, 1]); ax.set_yticks([]); ax.set_ylabel(name, rotation=0, ha="right", va="center")
axes[3].imshow(np.isfinite(Bl[:, 0])[None, :], aspect="auto", cmap="Greys_r", extent=[x[0], x[-1], 0, 1]); axes[3].set_yticks([]); axes[3].set_ylabel("ball visible", rotation=0, ha="right", va="center")
axes[4].plot(x, mind, lw=0.6); axes[4].set_ylim(0, 15); axes[4].set_ylabel("nearest player\nto ball (m)", rotation=0, ha="right", va="center"); axes[4].set_xlabel("video seconds")
fig.suptitle("Public 5-minute window — red=H (Barça) blue=A (Athletic) grey=D / none")
fig.tight_layout(); out = os.path.join(ROOT, "eda/out/timeline_public.png"); fig.savefig(out, dpi=110); print("wrote", out)
# a couple of numbers
seen = near >= 0
print("frames with ball: %.2f; nearest-player team agrees with GT (alive frames w/ ball): %.3f" % (seen.mean(), (near[seen] == np.vectorize(code.get)(gt)[seen]).mean()))
alive = gt != "D"; print("GT alive but ball missing: %.3f ; GT dead but ball visible: %.3f" % ((~np.isfinite(Bl[alive, 0])).mean(), np.isfinite(Bl[~alive, 0]).mean()))
