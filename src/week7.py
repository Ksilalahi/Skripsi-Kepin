from pathlib import Path
import pandas as pd


INPUT_PLAN = Path(
    "manifests/acquisition_plan.csv"
)

OUTPUT_PLAN = Path(
    "manifests/week7_real_replay_plan.csv"
)

RECORDING_DIR = Path(
    "data/real_replay/recordings_master"
)

PLAYBACK_DIR = Path(
    "data/real_replay/playback_master"
)


def main():

    print("=" * 50)
    print("MINGGU 7 - BUILD REAL REPLAY PLAN")
    print("=" * 50)

    if not INPUT_PLAN.exists():
        raise FileNotFoundError(
            f"Tidak ditemukan: {INPUT_PLAN}"
        )

    df = pd.read_csv(INPUT_PLAN)

    required = [
        "source_file_id",
        "file_path",
        "label",
        "room",
        "playback",
        "mic",
        "distance",
        "angle",
        "repetition"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Kolom acquisition plan tidak lengkap: "
            + ", ".join(missing)
        )

    # ========================================================
    # MINGGU 7 = REPETITION 1
    # ========================================================

    week7 = df[
        df["repetition"].astype(int) == 1
    ].copy()

    week7 = week7.reset_index(drop=True)

    print()
    print(
        "Total Minggu 7:",
        len(week7)
    )

    print(
        "Unique source:",
        week7["source_file_id"].nunique()
    )

    # ========================================================
    # VALIDASI TARGET
    # ========================================================

    if len(week7) != 240:
        raise RuntimeError(
            f"Expected 240 rows untuk repetition=1, "
            f"tetapi ditemukan {len(week7)}."
        )

    if week7["source_file_id"].nunique() != 120:
        raise RuntimeError(
            "Minggu 7 seharusnya mencakup "
            "120 source unik."
        )

    rows_per_source = (
        week7.groupby("source_file_id")
        .size()
    )

    if not (
        rows_per_source == 2
    ).all():
        raise RuntimeError(
            "Setiap source pada Minggu 7 "
            "harus memiliki tepat 2 recording."
        )

    # ========================================================
    # RECORDING ID
    # ========================================================

    week7["recording_id"] = [
        f"REAL_{i:04d}"
        for i in range(
            1,
            len(week7) + 1
        )
    ]

    # ========================================================
    # PLAYBACK MASTER
    # satu master dapat dipakai ulang untuk source sama
    # ========================================================

    week7["playback_path"] = (
        week7["source_file_id"]
        .apply(
            lambda x:
                str(
                    PLAYBACK_DIR
                    /
                    f"{x}.wav"
                )
        )
    )

    # ========================================================
    # OUTPUT MASTER RECORDING
    # ========================================================

    week7["output_path"] = (
        week7["recording_id"]
        .apply(
            lambda x:
                str(
                    RECORDING_DIR
                    /
                    f"{x}.wav"
                )
        )
    )

    # ========================================================
    # STATUS
    # ========================================================

    week7["recording_status"] = "pending"

    week7["qc_status"] = "NOT_CHECKED"

    # ========================================================
    # OPTIONAL SESSION METADATA
    # ========================================================

    week7["record_sr"] = 48000
    week7["record_subtype"] = "PCM_24"

    # Isi nanti sesuai kondisi aktual
    week7["speaker_level"] = ""
    week7["mic_level"] = ""
    week7["noise_floor"] = ""
    week7["agc_status"] = ""

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PLAN.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RECORDING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLAYBACK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    week7.to_csv(
        OUTPUT_PLAN,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 50)
    print("WEEK 7 PLAN SUMMARY")
    print("=" * 50)

    print(
        "Total recordings :",
        len(week7)
    )

    print(
        "Unique sources   :",
        week7[
            "source_file_id"
        ].nunique()
    )

    print()
    print("Label:")
    print(
        week7[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("Room:")
    print(
        week7[
            "room"
        ].value_counts()
    )

    print()
    print("Playback:")
    print(
        week7[
            "playback"
        ].value_counts()
    )

    print()
    print("Mic:")
    print(
        week7[
            "mic"
        ].value_counts()
    )

    print()
    print("Distance:")
    print(
        week7[
            "distance"
        ].value_counts()
    )

    print()
    print("Angle:")
    print(
        week7[
            "angle"
        ].value_counts()
    )

    print()
    print(
        "Saved:",
        OUTPUT_PLAN
    )

if __name__ == "__main__":
    main()