# External data sources — what exists, what it's for

Rules recap (RULES.md, FAQ.md): external data is allowed if it is about **other games**, public,
and documented in the tech doc (source, who labelled it, how much). **Nothing human-made about
BAR-ATH itself**: no ESPN / FotMob / Opta / Sofascore play-by-play, no corrected tracking, no
annotations of this match from anywhere. No data or extracts of the provided game may be sent to a
hosted LLM. Algorithmic pseudo-labels on BAR-ATH (smart tagging, our own detectors) are fine.

## Tier 1 — labels in our own label space (H / A / dead) on tracking

### DFL / IDSSE open dataset
- 7 full Bundesliga 1+2 matches 2022/23, 25 Hz tracking for all players + ball, official event
  data, **per-frame `possession` (Home/Away) and `ballstatus` (alive/dead) codes**. CC-BY 4.0.
  ~2.4 GB. Loader: `pip install floodlight` (already in `.venv`), then
  `from floodlight.io.datasets import IDSSEDataset; ev, xy, pos, bs, ts, pitch = IDSSEDataset().get("J03WMX")`
  (match ids in the floodlight docs; `IDSSEDataset("all")` for everything; data lands in the
  package's `.data/` folder).
- Use cases
  - **Train the tracking -> {H,A,D} model** on real labels: touch detection, nearest-player
    features, ball speed/direction change, team shape -> per-frame label. Then run it unchanged
    on BAR-ATH.
  - **Calibrate possession-change priors** for the HMM/Viterbi step: distribution of possession
    durations, switch rate per minute, dead-ball fraction and dead-ball durations, how long after a
    tackle the label flips. These replace tuning on the 5 public minutes.
  - **Learn what "dead ball" looks like in tracking**: ball-to-nearest-player distance, ball speed,
    player spread/compactness, ball off pitch, players walking to set-piece positions.
  - **Measure our scorer on labels we're allowed to look at**: run `evaluation/run_evaluation.py`
    logic on a DFL match with our pipeline's output to get a spellF1 that is honest.
  - Caveats: Bundesliga ≠ La Liga tempo; DFL ball tracking is near-complete while ours is 57%
    visible, so **degrade DFL ball data** (drop ball on a schedule matching our gap statistics)
    before training, or the model will trust the ball too much.

### Metrica sample-data (github.com/metrica-sports/sample-data)
- 3 anonymised games, tracking (25 Hz) + events, synchronised, 105x68, 0–1 normalised with origin
  top-left — the **same provider and coordinate convention as our ATD**. Game 3 is in EPTS/JSON
  (use kloppy). No licence beyond "be responsible, acknowledge the source".
- Use cases
  - **Derive team possession per frame from events** (passes, carries, recoveries, ball lost,
    set pieces all carry a team) -> second training/calibration set in the same coordinates.
  - **Sanity-check our loader and features** against data we know is clean.
  - **Dead-ball timing**: event data has set-piece and out-of-play events to learn restart
    patterns (how long between "ball out" and the throw-in).

## Tier 2 — broadcast-derived data (our broadcast ATD is the weak link)

### PFF FC 2022 World Cup (free with signup, blog.fc.pff.com)
- All 64 matches: broadcast-derived tracking + rich event data incl. possession events. Kloppy
  loads the tracking.
- Use cases: learn how **broadcast tracking gaps** (cutaways, replays, zooms) relate to game
  state; train a "which team has it" model that only sees the partial broadcast view; build the
  **broadcast -> tactical alignment** method on data where both views aren't available but
  possession is, to see how much broadcast alone recovers.

### SkillCorner open data (github.com/SkillCorner/opendata)
- 10 A-League matches of broadcast tracking + SkillCorner "Dynamic Events" and phases of play.
  No video, no raw events.
- Use cases: gap-behaviour stats for broadcast tracking; their phases-of-play labels are a
  reasonable proxy for possession spells for calibration.

## Tier 3 — commentary / audio

### SoccerNet-Echoes (github.com/SoccerNet/sn-echoes, arXiv 2405.07354)
- Whisper (v1/v2/v3) ASR transcripts of broadcast commentary for the SoccerNet games, JSON
  `{segments: {i: {start_time, end_time, text}}}` per half, with English translations. Transcripts
  are open on GitHub; the SoccerNet **video and action labels** need the SoccerNet NDA (free,
  academic-style form).
- Pairs with **SoccerNet Ball Action Spotting** (SN-BAS 2023/2024: pass, drive, header, high pass,
  out, cross, throw-in, shot, block, tackle, free kick, goal with timestamps).
- Use cases
  - **Train a commentary -> event / team model**: given a window of transcript, predict
    "possession changed", "ball out", "set piece", "shot" and which side. Then run it on our own
    Whisper output of the broadcast.
  - **Build a keyword/phrase lexicon** (player surnames -> team, "throw-in", "corner", "offside",
    "foul", "goal kick") with empirical reliability from Echoes rather than guessing.
  - **Calibrate commentary lag**: how many seconds after an event the commentator mentions it.
    Matters because our commentary timestamps need shifting before they become frame features.
  - Caveat: Echoes is itself Whisper output, so errors are correlated with ours — useful for
    robustness, not as perfect truth.

### GOAL (arXiv 2211.04534)
- ~20 games, raw audio + **human** transcripts of live commentary, grounded to video.
- Use cases: small clean set to measure our own Whisper + Demucs error rate on football
  commentary; whistle / crowd-roar detection positives.

## Tier 4 — our own match, algorithmic only (allowed)
- Smart-tagging XML (638 phases with team; set pieces; throw-ins have no team).
- Tactical video: run our own ball detector / re-id if the ATD's ball gaps matter; referee
  pose for whistle moments. Audio of the tactical mp4 is near-silent (-58 dB) — don't bother.
- Broadcast video: scoreboard OCR gives match clock -> **broadcast→tactical alignment for free**
  (the clock is in the picture; read it at a few hundred frames, fit a piecewise-linear map).
  Replays/cutaways = frames where the homography is invalid (34% of frames).

## How the pieces fit
```
DFL + Metrica  -> train/calibrate tracking->H/A/D model, HMM priors, dead-ball detector
PFF/SkillCorner-> how to use (or ignore) the broadcast ATD
Echoes + BAS   -> commentary->event model, lexicon, lag calibration
GOAL           -> Whisper/Demucs error rate, whistle detector positives
scoreboard OCR -> broadcast seconds -> tactical frame
```
Everything used must be listed in the tech doc with source, labeller and volume (SUBMISSION.md).

## Ideas queue (2026-10-25, in payoff order)
1. **Scoreboard OCR for broadcast→tactical alignment.** The broadcast picture has the match clock.
   OCR it (tesseract) on a few hundred sampled frames, fit a piecewise-linear map
   broadcast_frame -> tactical_frame (anchor: tactical kickoff frames 11185 / 113742 = clock 00:00 /
   45:00). Then every commentary timestamp becomes a submission frame. No tracking-shape matching.
2. **Degrade DFL ball data before training.** Our ball is visible 57% of frames, DFL's ~100%.
   Drop DFL ball samples with our gap-length distribution so the model learns player geometry for
   ball-missing frames instead of over-trusting the ball.
3. **Honest validation on DFL.** Run the whole pipeline on a DFL match, convert its possession +
   ballstatus codes to H/A/D, score with `evaluation/run_evaluation.py` logic. This is the spellF1
   we are allowed to tune on; the 5 public minutes are not.
4. **Dead-ball detector first, separately.** D segments are long and X is forward-filled, so D
   boundaries probably move spellF1 more than H-vs-A on contested frames. Features: smart-tagging
   set pieces, ball off pitch / far from everyone, player spread, ball speed ~0, whistle.
5. **Whistle detector on the Demucs crowd stem.** `no_vocals` stem carries the referee whistle
   (sharp 3–4 kHz tone, ~0.3–1 s). Spectral-peak detector -> stoppage/restart marks on the
   broadcast timeline, mapped to tactical frames via (1).
6. **Possession as segmentation, not classification.** Detect touches (ball speed / direction
   change next to a player), assign each touch a team, Viterbi over {H,A,D} with switch costs
   calibrated from DFL possession-duration stats. Optimises the segment-IoU structure the scorer
   actually uses.
7. Commentary -> features: player-surname lexicon (BAR vs ATH squads are public, non-match data),
   restart words (throw-in, corner, free kick, goal kick, offside), with lag calibrated on
   SoccerNet-Echoes. Gate by alignment confidence from (1).
