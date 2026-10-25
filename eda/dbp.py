"""A databallpy Game built straight from the numpy cache, bypassing its loaders and validators
(they reject this uncorrected export: ball >10 m off pitch, 46 track slots, EPTS period format).

    from eda.dbp import game
    g = game()                       # tactical; game("broadcast") for the other feed
    g.tracking_data.add_velocity(g.get_column_ids() + ["ball"], filter_type="savitzky_golay", max_velocity=50)
    from databallpy.visualize import plot_tracking_data, save_tracking_video
    plot_tracking_data(g, 11260, add_velocities=True, add_player_possession=True)

Conventions databallpy expects: metres with origin at pitch centre, columns home_<n>_x/_y and
away_<n>_x/_y where n is the track slot, period_id 1/2 on graded frames and -999 elsewhere.
ball_status is "alive" where the ball is tracked (placeholder -- our model should overwrite it).
player_possession = nearest tracked player when the ball is visible (same placeholder).
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from games import GAMES, GAME, half_video_blocks
from eda.atd import load

L, W = GAMES[GAME]["pitch"]
KICKOFF = pd.Timestamp("2024-08-24 19:30:00", tz="Europe/Madrid")   # only has to be a valid clock

def game(source="tactical"):
    from databallpy.game import Game
    from databallpy.schemas import TrackingData, EventData
    from databallpy.utils.constants import MISSING_INT
    d = load(source); P, T, B = d["players"], d["team"], d["ball"]; F = len(B)
    side = np.where(T == 0, "home", "away"); slot = np.arange(1, len(T) + 1)
    cols = {"frame": np.arange(F), "datetime": KICKOFF + pd.to_timedelta(np.arange(F) / 25, unit="s"),
            "period_id": np.full(F, MISSING_INT), "ball_x": B[:, 0] - L / 2, "ball_y": B[:, 1] - W / 2, "ball_z": np.zeros(F),
            "ball_status": np.where(np.isfinite(B[:, 0]), "alive", "dead"), "team_possession": np.full(F, None)}
    for s in range(len(T)):
        cols[f"{side[s]}_{slot[s]}_x"] = P[:, s, 0] - L / 2; cols[f"{side[s]}_{slot[s]}_y"] = P[:, s, 1] - W / 2
    dist = np.linalg.norm(P[:, :, :2] - B[:, None, :2], axis=2); seen = np.isfinite(dist).any(1)
    near = np.nanargmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    cols["player_possession"] = np.where(seen, np.char.add(np.char.add(side[near], "_"), slot[near].astype(str)), None)
    periods = []; gametime = np.full(F, "Break", dtype=object)
    if source == "tactical":
        for h, a, b in half_video_blocks(GAME):
            cols["period_id"][a:b + 1] = h
            sec = (np.arange(a, b + 1) - a) / 25 + (h - 1) * 45 * 60
            gametime[a:b + 1] = [f"{int(x // 60):02d}:{int(x % 60):02d}" for x in sec]
            periods.append({"period_id": h, "start_frame": a, "end_frame": b, "start_datetime_td": cols["datetime"][a], "end_datetime_td": cols["datetime"][b],
                            "start_datetime_ed": pd.NaT, "end_datetime_ed": pd.NaT})
    for h in range(len(periods) + 1, 6):
        periods.append({"period_id": h, "start_frame": MISSING_INT, "end_frame": MISSING_INT, "start_datetime_td": pd.NaT, "end_datetime_td": pd.NaT, "start_datetime_ed": pd.NaT, "end_datetime_ed": pd.NaT})
    cols["gametime_td"] = gametime
    td = TrackingData(pd.DataFrame(cols), provider="metrica", frame_rate=25)
    def players(mask):
        return pd.DataFrame({"id": slot[mask], "full_name": [f"Track {s}" for s in slot[mask]], "shirt_num": slot[mask], "position": "unspecified",
                             "start_frame": MISSING_INT, "end_frame": MISSING_INT, "starter": True})
    return Game(tracking_data=td, event_data=EventData(pd.DataFrame()), pitch_dimensions=[L, W], periods=pd.DataFrame(periods),
                home_team_id=1, home_team_name="Barcelona", home_players=players(T == 0), home_score=MISSING_INT, home_formation="",
                away_team_id=2, away_team_name="Atletic Club", away_players=players(T == 1), away_score=MISSING_INT, away_formation="",
                country="Spain", shot_events=pd.DataFrame(), dribble_events=pd.DataFrame(), pass_events=pd.DataFrame(), _check_inputs_=False)

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    g = game(); td = g.tracking_data
    print(td.shape, "period_id counts", td.period_id.value_counts().to_dict())
    print("column ids:", g.get_column_ids()[:3], "...", len(g.get_column_ids()))
    td.add_velocity(g.get_column_ids() + ["ball"], filter_type="savitzky_golay", max_velocity=50.0)
    print("velocity cols added:", [c for c in td.columns if c.endswith("_vx")][:3])
    from databallpy.visualize import plot_tracking_data, save_tracking_video
    fig, ax = plot_tracking_data(g, 11260, add_velocities=True, add_player_possession=True, title="databallpy plot_tracking_data")
    fig.savefig(os.path.join(ROOT, "eda/out/dbp_frame_11260.png"), dpi=100); print("wrote eda/out/dbp_frame_11260.png")
    pc = td.get_pitch_control([L, W], start_idx=11260, end_idx=11262)
    print("pitch control:", None if pc is None else np.asarray(pc).shape)
    save_tracking_video(g, 11185, 11285, os.path.join(ROOT, "eda/out"), title="dbp_clip_11185_11285", add_velocities=True, add_player_possession=True, verbose=False)
    print("wrote eda/out/dbp_clip_11185_11285.mp4")
