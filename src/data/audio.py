from __future__ import annotations

from pathlib import Path

import soundfile as sf


def read_audio_info(audio_path: str | Path) -> dict[str, float | int]:
    info = sf.info(str(audio_path))
    return {
        "sample_rate": int(info.samplerate),
        "num_frames": int(info.frames),
        "duration_seconds": float(info.duration),
        "num_channels": int(info.channels),
    }

