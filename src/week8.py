from pathlib import Path
import pandas as pd


INPUT_PLAN = Path("manifests/acquisition_plan.csv")
OUTPUT_PLAN = Path("manifests/week8_real_replay_plan.csv")

PLAYBACK_DIR = Path("data/real_replay/playback_master")
RECORDING_DIR = Path("data/real_replay/recordings_master")


def main():
    if not INPUT_PLAN.exists():
        raise FileNotFoundError(
            f"Acquisition plan tidak ditemukan: {INPUT_PLAN}"
        )

    df = pd.read_csv(INPUT_PLAN)

    print("Total acquisition plan :", len(df))

    if "repetition" not in df.columns:
        raise ValueError("Kolom 'repetition' tidak ditemukan.")

    # Week 8 = repetition kedua
    week8 = df[df["repetition"] == 2].copy()
    week8 = week8.reset_index(drop=True)

    print("Jumlah repetition 2   :", len(week8))
    print("Unique source         :", week8["source_file_id"].nunique())

    # Validasi sesuai rancangan penelitian
    if len(week8) != 240:
        raise ValueError(
            f"Week 8 seharusnya 240 recording, "
            f"tetapi ditemukan {len(week8)}."
        )

    if week8["source_file_id"].nunique() != 120:
        raise ValueError(
            "Week 8 seharusnya memiliki 120 source unik."
        )

    counts = week8.groupby("source_file_id").size()

    if not (counts == 2).all():
        bad = counts[counts != 2]

        raise ValueError(
            "Setiap source harus memiliki tepat 2 konfigurasi.\n"
            f"{bad}"
        )

    # REAL_0241 - REAL_0480
    week8["recording_id"] = [
        f"REAL_{i:04d}"
        for i in range(241, 481)
    ]

    # Playback master sama seperti Week 7
    week8["playback_path"] = week8["source_file_id"].apply(
        lambda x: str(
            PLAYBACK_DIR / f"{x}.wav"
        )
    )

    # File recording baru
    week8["recording_path"] = week8["recording_id"].apply(
        lambda x: str(
            RECORDING_DIR / f"{x}.wav"
        )
    )

    # Status recording
    week8["recording_status"] = "pending"
    week8["record_error"] = ""

    # Recording format
    week8["record_sr"] = 48000
    week8["record_subtype"] = "PCM_24"

    # Setting yang sudah stabil pada Week 7
    week8["speaker_level"] = 50.0
    week8["mic_level"] = 70.0

    # Isi sesuai kondisi nyata
    week8["noise_floor"] = ""
    week8["agc_status"] = "off"

    # Recording statistics
    week8["record_peak"] = ""
    week8["record_rms"] = ""
    week8["record_clip_ratio"] = ""
    week8["record_duration_s"] = ""

    # QC
    week8["qc_status"] = "NOT_CHECKED"
    week8["qc_reason"] = ""
    week8["qc_sr"] = ""
    week8["qc_duration_s"] = ""
    week8["qc_peak"] = ""
    week8["qc_rms"] = ""
    week8["qc_clip_ratio"] = ""
    week8["qc_finite"] = ""

    OUTPUT_PLAN.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    week8.to_csv(
        OUTPUT_PLAN,
        index=False
    )

    print()
    print("===================================")
    print("WEEK 8 PLAN BERHASIL DIBUAT")
    print("===================================")
    print("Output             :", OUTPUT_PLAN)
    print("Recording          :", len(week8))
    print(
        "Recording ID       :",
        week8.iloc[0]["recording_id"],
        "-",
        week8.iloc[-1]["recording_id"]
    )
    print(
        "Unique recording   :",
        week8["recording_id"].nunique()
    )
    print(
        "Unique source      :",
        week8["source_file_id"].nunique()
    )
    print()
    print("Repetition:")
    print(
        week8["repetition"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Label:")
    print(
        week8["label"]
        .value_counts()
        .sort_index()
    )

    if "room" in week8.columns:
        print()
        print("Room:")
        print(
            week8["room"]
            .value_counts()
            .sort_index()
        )

    if "mic" in week8.columns:
        print()
        print("Mic:")
        print(
            week8["mic"]
            .value_counts()
        )

    if "playback" in week8.columns:
        print()
        print("Playback:")
        print(
            week8["playback"]
            .value_counts()
        )

if __name__ == "__main__":
    main()