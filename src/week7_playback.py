from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from scipy.signal import resample_poly

PLAN_PATH = Path("manifests/week7_real_replay_plan.csv")
TARGET_SR = 48000

def prepare_master(
    input_path,
    output_path
):

    audio, sr = sf.read(
        input_path,
        always_2d=True,
        dtype="float32"
    )

    # mono
    audio = audio.mean(
        axis=1
    )

    if sr != TARGET_SR:

        audio = resample_poly(
            audio,
            TARGET_SR,
            sr
        )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # conservative normalization
    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:

        audio = (
            audio / peak
        ) * 0.8

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        output_path,
        audio,
        TARGET_SR,
        subtype="PCM_24"
    )


def main():

    if not PLAN_PATH.exists():
        raise FileNotFoundError(
            PLAN_PATH
        )

    df = pd.read_csv(
        PLAN_PATH
    )

    unique_sources = (
        df[
            [
                "source_file_id",
                "file_path",
                "playback_path"
            ]
        ]
        .drop_duplicates(
            subset=[
                "source_file_id"
            ]
        )
        .reset_index(drop=True)
    )

    print("=" * 50)
    print("PREPARE WEEK 7 PLAYBACK MASTER")
    print("=" * 50)

    print(
        "Unique source:",
        len(unique_sources)
    )

    for i, row in (
        unique_sources.iterrows()
    ):

        input_path = row[
            "file_path"
        ]

        output_path = row[
            "playback_path"
        ]

        if Path(
            output_path
        ).exists():

            print(
                f"[{i+1:03d}/"
                f"{len(unique_sources):03d}] "
                f"SKIP "
                f"{row['source_file_id']}"
            )

            continue

        prepare_master(
            input_path,
            output_path
        )

        print(
            f"[{i+1:03d}/"
            f"{len(unique_sources):03d}] "
            f"OK "
            f"{row['source_file_id']}"
        )

    print()
    print(
        "Playback master selesai."
    )


if __name__ == "__main__":
    main()