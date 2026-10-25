# MTAH 2027 — team notes (Elijah, Oct 25)

## The task in one line
One LaLiga match (Barcelona vs Athletic, 2024-25 MD2). For each of the 199,319 frames of the
tactical video (25 fps) output `H` (Barça has ball), `A` (Athletic has ball), `D` (dead ball),
or `X` (abstain). Scored on the private full-match truth; we only see the first 5 min of play.

## Scoring — this drives everything
- Ranked on **spellF1**: each possession (maximal run of H or A) is one object. Predicted
  possessions are matched 1:1 to true ones by temporal IoU at thresholds 0.3..0.7, F1 averaged.
  Constant guess = 0. A possession 1 s late scores 0.65 on its own. Noise band ~0.04.
- **macroF1** (per-frame, mean of H/A/D F1) is the tie-break only.
- `X` is forward-filled with our previous label before spellF1 runs. So X inside a possession is
  free; X across a hand-over is as bad as being wrong.
- Baseline (nearest player to ball + 2 s majority vote) = spellF1 0.2433 full match. Beat that.
- Top 20 advances. Deadline Nov 17. Deliverables: CSV + public repo that regenerates it + short
  technical doc.

## Rules that bite
- No hand labels on this game, for anything (train, validate, fix). Pseudo-labels from an
  algorithm are fine. Hand-labelled OTHER games are fine.
- No event data / corrected tracking about this game from anywhere.
- **Do not upload the video or data to hosted LLM APIs** (terms of use). Local models only
  (whisper via mlx, local YOLO, etc). This also rules out Polyglot's Claude translate backend.
- Public submission repo must not contain data/ or derived frame-level exports.

## The data (all in `data/`, numbers verified locally)
| asset | what | notes |
|---|---|---|
| tactical ATD (`_atd.txt` + `.xml`) | 199,319 lines, `frame:p1;p2;..;p46:ball`, each `x,y,speed`, 0-1 normalised | row i = video frame i-1. 46 track *slots*, not players; slots 1-23 Barça, 24-46 Athletic (teamId in xml). Ball visible 56.9% all / 59.3% graded. ~20.5 players tracked per graded frame |
| tactical homography | 3x3 pitch(0-1)->pixel per frame, 100% valid | lets us project tracking onto the video |
| smart tagging xml | 638 phases (`BUILD UP`, `SET PIECES`...), seconds from video start, with `Team` tag | `Team` = N/A on all throw-ins. Phases, not possessions |
| broadcast ATD + homography | 164,488 frames, own timeline, 65.8% valid homography | not synced to tactical; has the commentary audio |
| videos | tactical 199,319 f; broadcast 164,488 f; ~13 GB | downloading into `data_raw/` |

Graded frames: 11185–90651 (1st half) and 110808–190197 (2nd half). 2nd half kicks off at
113742 — the 2,934 frames before that are graded and all `D`.

## What the public 5 minutes shows (`eda/out/timeline_public.png`)
- 21 segments / 17 possessions, mean 14 s, but possessions as short as **0.76 s and 1.3 s** exist
  (GT frames 15346-15364, 15766-15798). A 2 s majority vote erases those by construction.
- Nearest-player team agrees with GT on only 66% of alive frames with a ball: every pass flips it.
- Dead ball ≠ missing ball: in this window the ball is *visible* on 66% of D frames, and missing
  on 6.5% of alive frames. The baseline's "no ball → D" is the wrong proxy.
- Dead ball looks like: ball far from every player (nearest-player distance 10-15 m) and/or ball
  outside the pitch rectangle, with players walking. Smart tagging `SET PIECES` marks the restart.
- Smart tagging `Team` is roughly right at phase level but lags transitions by seconds and is
  blank during set-piece build-ups.

## Plan (proposal, individual week then converge)
1. **Feature table per frame** (numpy, cached in `eda/atd_tactical.npz` already): ball xy/speed,
   nearest player per team + distance, ball in/out of pitch, ball visible, smart-tag phase/team,
   n tracked players, team centroids / spread, ball velocity direction vs team.
2. **Dead-ball detector first** (D vs alive). It's the biggest easy win: ball off pitch, players
   static, SET PIECES tag, 2nd-half pre-kickoff. Whistle detection from audio if tactical video
   has audio (check `ffprobe` when the mp4 lands).
3. **Possession as a segmentation problem, not a per-frame classifier.** Detect *touch events*
   (ball speed/direction change near a player), assign each touch a team, then a possession
   changes when the opposing team gets N consecutive touches / holds for T seconds. Tune N,T on
   the whole-match *structure* (possession count, duration distribution look like football), not
   on the 5 minutes. HMM / Viterbi over {H,A,D} with touch-based emissions is the clean version.
4. **Ball gap filling**: interpolate short gaps with constant velocity; for long gaps, the player
   cluster motion tells you where play is.
5. Only then: broadcast video + commentary (Polyglot's mlx-whisper + Demucs path, local), needs
   a broadcast→tactical time alignment (match on player-cluster shape through both homographies).

## Commentary (Polyglot, local only)
Broadcast mp4 audio -> ffmpeg 16 kHz wav -> Demucs two-stems (vocals = commentators, no_vocals =
crowd + whistle) -> mlx-whisper large-v3-turbo with word timestamps -> `eda/audio/commentary_*.jsonl`.
Times are broadcast seconds; mapping to tactical frames needs the broadcast->tactical alignment.
Only Polyglot's mlx-whisper + Demucs path is allowed; its Gemini-audio and Claude backends upload
the audio and are off limits for this data.

## databallpy
`eda/dbp.py` builds a real databallpy `Game` from the numpy cache (its own loaders reject this
export). Everything on https://databallpy.readthedocs.io/en/latest/getting_started/visualizations_page.html
then works: `plot_tracking_data`, `save_tracking_video`, `add_velocity`, `get_pitch_control`.
Placeholders to replace with our model's output: `ball_status` (= ball tracked) and
`player_possession` (= nearest player). Team colours default to green=home, red=away; pass
`team_colors=["#d62728", "#1f77b4"]` to match `eda/render.py`.
```python
from eda.dbp import game
g = game(); td = g.tracking_data
td.add_velocity(g.get_column_ids() + ["ball"], filter_type="savitzky_golay", max_velocity=50)
from databallpy.visualize import plot_tracking_data, save_tracking_video
plot_tracking_data(g, 11260, add_velocities=True, add_player_possession=True)
save_tracking_video(g, 11185, 11685, "eda/out", title="clip", add_velocities=True, add_player_possession=True)
```

## Videos
`data/tactical/BAR-ATH_tactical.mp4` (5.2 GB, from the team bucket `s3://more-than-a-hack-2027`,
profile `mtah`). 199,319 frames at 25 fps, **has a stereo AAC audio track** -> whistle / crowd
detection on the submission timeline with no alignment problem. Broadcast mp4 not uploaded yet.

## Running things
```
.venv/bin/python eda/atd.py                 # builds/validates the numpy cache (5 s)
.venv/bin/python eda/plot.py 11185 15889    # pitch PNGs -> eda/out/
.venv/bin/python eda/plot.py --clip 11185 11600   # GIF
.venv/bin/python eda/timeline.py            # the 5-min comparison figure
.venv/bin/python eda/render.py --source tactical  --start 11185 --end 11685          # pitch-view mp4
.venv/bin/python eda/render.py --source tactical --start 11185 --end 11685 --video --possession --trail   # camera + overlay | pitch
.venv/bin/python eda/render.py --source tactical --start 11185 --end 18685 --velocities --possession --trail --heatmap   # all overlays
.venv/bin/python eda/dbp.py                 # databallpy smoke test -> eda/out/dbp_*
~/Polyglot/.venv/bin/python eda/commentary.py --start 600 --end 900                  # local whisper on broadcast audio
.venv/bin/python starter/baseline.py        # regenerate the 4 baselines
.venv/bin/python evaluation/run_evaluation.py   # score everything in evaluation/results/
```
