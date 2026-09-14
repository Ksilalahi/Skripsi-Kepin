from pathlib import Path

import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

PLAN = "manifests/pilot_acquisition_plan.csv"
OUT_DIR = Path("data/real_replay/playback_master")
TARGET_SR = 48000

def main():
    df = pd.read_csv(PLAN)
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    playback_paths = []
    for _, row in df.iterrows():
        input_path = Path(
            row["file_path"]
        )

        pilot_id = row["pilot_id"]
        output_path = (
            OUT_DIR /
            f"{pilot_id}.wav"
        )

        audio, sr = sf.read(
            input_path,
            always_2d=True
        )

        # mono
        audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = resample_poly(
                audio,
                TARGET_SR,
                sr
            )

        peak = abs(audio).max()

        if peak > 1e-6:
            audio = audio / peak * 0.8

        sf.write(
            output_path,
            audio,
            TARGET_SR,
            subtype="PCM_24"
        )

        playback_paths.append(
            str(output_path)
        )
        print(f"{pilot_id} -> {output_path}")

    df["playback_path"] = playback_paths
    df.to_csv(PLAN,index=False)
    print("\nSemua playback master selesai.")

if __name__ == "__main__":
    main()