from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.audio import read_audio_info
from src.data.split import assign_split


MODALITY_MAP = {
    "01": "full_av",
    "02": "video_only",
    "03": "audio_only",
}

CHANNEL_MAP = {
    "01": "speech",
    "02": "song",
}

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}

STATEMENT_MAP = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}

REPETITION_MAP = {
    "01": "first",
    "02": "second",
}

FINAL_LABEL_MAP = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fearful",
    "disgust": "disgust",
}


def infer_actor_gender(actor_id: str) -> str:
    actor_num = int(actor_id)
    return "male" if actor_num % 2 else "female"


def parse_ravdess_filename(audio_path: str | Path) -> dict[str, str]:
    audio_path = Path(audio_path)
    stem_parts = audio_path.stem.split("-")
    if len(stem_parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename format: {audio_path.name}")

    modality_code, channel_code, emotion_code, intensity_code, statement_code, repetition_code, actor_code = stem_parts

    emotion = EMOTION_MAP[emotion_code]
    final_label = FINAL_LABEL_MAP.get(emotion)

    parsed = {
        "file_name": audio_path.name,
        "file_path": str(audio_path.resolve()),
        "modality_code": modality_code,
        "modality": MODALITY_MAP[modality_code],
        "channel_code": channel_code,
        "channel": CHANNEL_MAP[channel_code],
        "emotion_code": emotion_code,
        "emotion": emotion,
        "intensity_code": intensity_code,
        "intensity": INTENSITY_MAP[intensity_code],
        "statement_code": statement_code,
        "statement": STATEMENT_MAP[statement_code],
        "repetition_code": repetition_code,
        "repetition": REPETITION_MAP[repetition_code],
        "actor_id": actor_code,
        "actor_gender": infer_actor_gender(actor_code),
        "final_label": final_label,
        "keep_for_project": final_label is not None,
    }
    return parsed


def build_ravdess_metadata(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    wav_paths = sorted(raw_dir.rglob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(
            f"No .wav files found under {raw_dir}. Place the RAVDESS speech audio files under data/raw first."
        )

    records: list[dict[str, object]] = []
    for wav_path in wav_paths:
        parsed = parse_ravdess_filename(wav_path)
        if parsed["modality"] != "audio_only":
            continue
        if parsed["channel"] != "speech":
            continue

        audio_info = read_audio_info(wav_path)
        split = assign_split(str(parsed["actor_id"]))
        parsed.update(audio_info)
        parsed["split"] = split
        records.append(parsed)

    if not records:
        raise FileNotFoundError(
            f"Found .wav files under {raw_dir}, but none matched the audio-only speech subset."
        )

    df = pd.DataFrame(records).sort_values(["actor_id", "emotion_code", "statement_code", "repetition_code"])
    return df.reset_index(drop=True)


def save_metadata(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAVDESS speech metadata table.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing raw RAVDESS files.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV or parquet path.")
    args = parser.parse_args()

    df = build_ravdess_metadata(args.raw_dir)
    save_metadata(df, args.output)
    kept = int(df["keep_for_project"].sum())
    print(f"Saved metadata for {len(df)} speech clips to {args.output}")
    print(f"{kept} clips are included in the 6-class project label schema.")


if __name__ == "__main__":
    main()

