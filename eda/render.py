"""Render tracking as a video. Pitch view always; add --video to put the real camera frame beside
it with the tracking projected onto it through the homography (needs the mp4 in data/).

    .venv/bin/python eda/render.py --source tactical  --start 11185 --end 11600
    .venv/bin/python eda/render.py --source broadcast --start 20000 --end 20500 --video
    --step 2 renders every 2nd frame (half the work, same wall-clock speed in the output)
    --velocities --possession --trail --heatmap   databallpy-style overlays (heatmap = pitch control, slow)

Frames are the source's own video frames (tactical = submission rows). The two sources are NOT
time-aligned, so the same --start means different moments of the match.
"""
import argparse, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from games import asset_paths, GAMES, GAME
from starter.utils import draw_pitch, HOME_COLOUR, AWAY_COLOUR
from eda.atd import load

L, W = GAMES[GAME]["pitch"]
RGB = {0: (214, 39, 40), 1: (31, 119, 180)}; RGB_HEX = {0: HOME_COLOUR, 1: AWAY_COLOUR}
GT = np.loadtxt(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"), dtype=str, delimiter=",", skiprows=1, usecols=1)

def velocity(d, i, lag=5):
    """m/s from a 0.2 s position difference. NaN where either end is untracked."""
    j = max(i - lag, 0); return (d["players"][i, :, :2] - d["players"][j, :, :2]) * (25 / max(i - j, 1))

def pitch_control(d, i, v, nx=53, ny=34):
    """Fernandez & Born (2018) surface via databallpy's own function: 1 = home control, 0 = away."""
    import pandas as pd
    from databallpy.features.pitch_control import get_pitch_control_single_frame
    p, t, b = d["players"][i], d["team"], d["ball"][i]
    row = {"ball_x": b[0] - L / 2, "ball_y": b[1] - W / 2}
    for s in np.where(np.isfinite(p[:, 0]) & np.isfinite(v[:, 0]))[0]:
        k = f"{'home' if t[s] == 0 else 'away'}_{s}"
        row.update({f"{k}_x": p[s, 0] - L / 2, f"{k}_y": p[s, 1] - W / 2, f"{k}_vx": v[s, 0], f"{k}_vy": v[s, 1]})
    if not np.isfinite(b[0]): row["ball_x"], row["ball_y"] = 0.0, 0.0   # ponytail: no ball -> centre; distance-to-ball term is wrong on those frames
    return get_pitch_control_single_frame(pd.Series(row), [L, W], nx, ny)

def pitch_panel(d, i, fig, ax, arts, source, opts):
    p, t, b = d["players"][i], d["team"], d["ball"][i]
    for side in (0, 1):
        m = (t == side) & np.isfinite(p[:, 0]); arts[side].set_offsets(p[m, :2])
    arts[2].set_offsets(b[None, :2] if np.isfinite(b[0]) else np.empty((0, 2)))
    v = velocity(d, i)
    if opts.velocities:
        m = np.isfinite(v[:, 0]) & np.isfinite(p[:, 0]); arts[3].remove()   # quiver cannot change arrow count in place
        arts[3] = ax.quiver(p[m, 0], p[m, 1], v[m, 0], v[m, 1], angles="xy", scale_units="xy", scale=1, width=0.003, color="k", zorder=2)
    if opts.possession:                       # ring on the nearest tracked player when the ball is seen
        dist = np.linalg.norm(p[:, :2] - b[:2], axis=1)
        if np.isfinite(dist).any():
            s = np.nanargmin(dist); arts[4].set_offsets(p[s, :2][None]); arts[4].set_edgecolor(RGB_HEX[t[s]]); arts[4].set_facecolor("none")
        else: arts[4].set_offsets(np.empty((0, 2)))
    if opts.heatmap:
        arts[5].set_data(pitch_control(d, i, v))
    if opts.trail:
        tr = d["ball"][max(i - 25, 0):i + 1, :2]; arts[6].set_data(tr[:, 0], tr[:, 1])
    gt = f"  gt={GT[i]}" if source == "tactical" else ""
    ax.set_title(f"{source} frame {i}  t={i/25:.1f}s{gt}  ball={'yes' if np.isfinite(b[0]) else 'MISSING'}  tracked={np.isfinite(p[:,0]).sum()}")
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

def video_frames(path, start, n, w, h):
    """Yield n RGB frames from video frame `start`, scaled to w x h. Accurate seek, one decode."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start/25:.6f}", "-i", path, "-frames:v", str(n),
           "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for _ in range(n):
        buf = proc.stdout.read(w * h * 3)
        if len(buf) < w * h * 3: break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.stdout.close(); proc.wait()

def overlay(img, d, i, H, scale):
    """Draw tracked positions on the camera frame. H maps normalised pitch (0-1) -> 1920x1080 px."""
    if not np.isfinite(H).all(): return img
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    def px(x, y):
        v = H @ np.array([x / L, y / W, 1.0]); return v[0] / v[2] * scale, v[1] / v[2] * scale
    p, t, b = d["players"][i], d["team"], d["ball"][i]
    for s in np.where(np.isfinite(p[:, 0]))[0]:
        u, v = px(p[s, 0], p[s, 1]); dr.ellipse([u-7, v-7, u+7, v+7], outline=RGB[t[s]], width=3)
    if np.isfinite(b[0]):
        u, v = px(b[0], b[1]); dr.ellipse([u-6, v-6, u+6, v+6], fill=(255, 255, 255), outline=(0, 0, 0))
    return np.asarray(im)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="tactical"); ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True); ap.add_argument("--step", type=int, default=1); ap.add_argument("--video", action="store_true"); ap.add_argument("--out")
    ap.add_argument("--velocities", action="store_true"); ap.add_argument("--possession", action="store_true")
    ap.add_argument("--heatmap", action="store_true", help="pitch-control surface (databallpy, Fernandez & Born)"); ap.add_argument("--trail", action="store_true")
    a = ap.parse_args()
    d = load(a.source); frames = list(range(a.start, a.end, a.step))
    out = a.out or os.path.join(ROOT, "eda/out", f"{a.source}_{a.start}_{a.end}{'_video' if a.video else ''}.mp4")
    fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=100); draw_pitch(ax, L, W)
    arts = [ax.scatter([], [], c=HOME_COLOUR, s=90, edgecolors="k", zorder=3), ax.scatter([], [], c=AWAY_COLOUR, s=90, edgecolors="k", zorder=3),
            ax.scatter([], [], c="w", s=50, edgecolors="k", zorder=4),
            ax.quiver(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), angles="xy", scale_units="xy", scale=1, width=0.003, color="k", zorder=2),
            ax.scatter([], [], s=420, facecolors="none", linewidths=2.5, zorder=2.5),
            ax.imshow(np.full((34, 53), 0.5), extent=[0, L, 0, W], origin="lower", cmap="coolwarm_r", vmin=0, vmax=1, alpha=0.45 if a.heatmap else 0, zorder=0.5),
            ax.plot([], [], color="w", lw=1.5, alpha=0.7, zorder=3.5)[0]]
    fig.tight_layout()
    if a.video:
        paths = asset_paths(source=a.source); vw, vh = 960, 540
        hom = np.loadtxt(paths["homography"], delimiter=",", skiprows=1)
        Hs = hom[:, 1:10].reshape(-1, 3, 3)
        vid = video_frames(paths["video"], a.start, a.end - a.start, vw, vh)
    ff = None
    for k, i in enumerate(frames):
        panel = pitch_panel(d, i, fig, ax, arts, a.source, a)
        if a.video:
            img = None
            for j in range(a.step):               # consume `step` decoded frames, keep the first
                f = next(vid, None); img = f if j == 0 else img
            if img is None: break
            cam = overlay(img, d, i, Hs[i], vw / 1920)
            ph = cam.shape[0]; panel = np.asarray(Image.fromarray(panel).resize((int(panel.shape[1] * ph / panel.shape[0]), ph)))
            panel = np.hstack([cam, panel])
        if ff is None:
            h, w = panel.shape[:2]; w -= w % 2; h -= h % 2
            ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(25 / a.step),
                                   "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], stdin=subprocess.PIPE)
        ff.stdin.write(np.ascontiguousarray(panel[:h, :w]).tobytes())
        if k % 100 == 0: print(f"{k}/{len(frames)}", flush=True)
    ff.stdin.close(); ff.wait(); print("wrote", out)

if __name__ == "__main__":
    main()
