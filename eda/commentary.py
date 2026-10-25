"""Broadcast commentary -> timestamped text, fully local, via Polyglot's Demucs + mlx-whisper.

    ~/Polyglot/.venv/bin/python eda/commentary.py                 # whole match
    ~/Polyglot/.venv/bin/python eda/commentary.py --no-separate   # skip Demucs (faster, noisier)
    ... --start 600 --end 900                                     # seconds, for a quick test

Writes eda/audio/broadcast.wav, the Demucs stems (vocals = commentators, no_vocals = crowd +
whistle), and eda/audio/commentary.jsonl with {start,end,text,words:[{word,start,end}]} per
segment. Times are BROADCAST seconds, not tactical: aligning to tactical frames is a separate step.

Allowed because everything runs on this machine. Do NOT switch Polyglot to its gemini-audio or
Claude backends for this data -- the competition terms forbid sending it to hosted LLMs.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
POLYGLOT = Path.home() / "Polyglot"; sys.path.insert(0, str(POLYGLOT / "src"))
from games import asset_paths

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-separate", action="store_true")
    ap.add_argument("--start", type=float); ap.add_argument("--end", type=float); ap.add_argument("--source", default="broadcast")
    a = ap.parse_args()
    out = ROOT / "eda" / "audio"; out.mkdir(exist_ok=True)
    tag = f"{a.source}" + (f"_{int(a.start)}_{int(a.end)}" if a.start is not None else "")
    wav = out / f"{tag}.wav"
    if not wav.exists():
        cut = (["-ss", str(a.start)] if a.start is not None else []) + (["-to", str(a.end)] if a.end is not None else [])
        subprocess.run(["ffmpeg", "-v", "error", "-y", *cut, "-i", asset_paths(source=a.source)["video"],
                        "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    speech = wav
    if not a.no_separate:
        from polyglot.config import load_settings
        from polyglot.separate import separate
        speech, _crowd = separate(wav, out / "stems", load_settings(POLYGLOT / "config" / "settings.toml"))
    import mlx_whisper
    res = mlx_whisper.transcribe(str(speech), path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                 word_timestamps=True, language=None)
    off = a.start or 0.0
    with open(out / f"commentary_{tag}.jsonl", "w") as fh:
        for s in res["segments"]:
            fh.write(json.dumps({"start": s["start"] + off, "end": s["end"] + off, "text": s["text"].strip(),
                                 "words": [{"word": w["word"], "start": w["start"] + off, "end": w["end"] + off} for w in s.get("words", [])]}) + "\n")
    print("language:", res.get("language"), "| segments:", len(res["segments"]), "->", out / f"commentary_{tag}.jsonl")
    for s in res["segments"][:5]: print(f"  {s['start']+off:7.1f}s  {s['text'].strip()}")

if __name__ == "__main__":
    main()
