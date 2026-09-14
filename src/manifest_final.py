from pathlib import Path
import pandas as pd

WEEK7_MANIFEST = Path(
    "manifests/real_replay_manifest_240.csv"
)

WEEK8_MANIFEST = Path(
    "manifests/week8_real_replay_plan.csv"
)

OUTPUT_MANIFEST = Path(
    "manifests/real_replay_manifest_480.csv"
)

def clean_status(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

def validate_required_columns(df, name):
    required = [
        "recording_id",
        "source_file_id",
        "label",
        "repetition",
        "recording_status",
        "qc_status",
        "record_sr",
        "record_subtype",
        "recording_path",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} kehilangan kolom: {missing}"
        )

def main():
    print()
    print("==============================================")
    print("MEMBUAT FINAL REAL REPLAY MANIFEST 480")
    print("==============================================")
    print()

    if not WEEK7_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest Week 7 tidak ditemukan: "
            f"{WEEK7_MANIFEST}"
        )

    if not WEEK8_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest Week 8 tidak ditemukan: "
            f"{WEEK8_MANIFEST}"
        )

    week7 = pd.read_csv(
        WEEK7_MANIFEST
    )

    week8 = pd.read_csv(
        WEEK8_MANIFEST
    )

    # ============================================================
    # SAMAKAN NAMA KOLOM WEEK 7 DAN WEEK 8
    # ============================================================

    if "output_path" in week7.columns and "recording_path" not in week7.columns:
        week7 = week7.rename(
            columns={
                "output_path": "recording_path"
        }
    )

    print("Kolom path Week 7 diseragamkan menjadi recording_path.")
    print(
        "Week 7 rows:",
        len(week7)
    )

    print(
        "Week 8 rows:",
        len(week8)
    )

    # --------------------------------------------------------
    # VALIDATE COLUMNS
    # --------------------------------------------------------

    validate_required_columns(
        week7,
        "Week 7"
    )

    validate_required_columns(
        week8,
        "Week 8"
    )

    # --------------------------------------------------------
    # FILTER VALID WEEK 7
    # --------------------------------------------------------

    week7_recording_status = clean_status(
        week7["recording_status"]
    )

    week7_qc_status = (
        week7["qc_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    week7_valid = week7[
        (week7_recording_status == "done")
        &
        (week7_qc_status == "PASS")
    ].copy()

    # --------------------------------------------------------
    # FILTER VALID WEEK 8
    # --------------------------------------------------------

    week8_recording_status = clean_status(
        week8["recording_status"]
    )

    week8_qc_status = (
        week8["qc_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    week8_valid = week8[
        (week8_recording_status == "done")
        &
        (week8_qc_status == "PASS")
    ].copy()

    print()
    print(
        "Week 7 valid:",
        len(week7_valid)
    )

    print(
        "Week 8 valid:",
        len(week8_valid)
    )

    # --------------------------------------------------------
    # STRICT CHECK BEFORE MERGE
    # --------------------------------------------------------

    if len(week7_valid) != 240:
        raise ValueError(
            f"Week 7 harus memiliki 240 recording "
            f"DONE + PASS, tetapi ditemukan "
            f"{len(week7_valid)}."
        )

    if len(week8_valid) != 240:
        raise ValueError(
            f"Week 8 harus memiliki 240 recording "
            f"DONE + PASS, tetapi ditemukan "
            f"{len(week8_valid)}."
        )

    if not (
        week7_valid["repetition"] == 1
    ).all():
        raise ValueError(
            "Manifest Week 7 tidak seluruhnya "
            "repetition = 1."
        )

    if not (
        week8_valid["repetition"] == 2
    ).all():
        raise ValueError(
            "Manifest Week 8 tidak seluruhnya "
            "repetition = 2."
        )

    # --------------------------------------------------------
    # ALIGN COLUMNS
    # --------------------------------------------------------

    all_columns = list(
        dict.fromkeys(
            list(week7_valid.columns)
            +
            list(week8_valid.columns)
        )
    )

    week7_valid = week7_valid.reindex(
        columns=all_columns
    )

    week8_valid = week8_valid.reindex(
        columns=all_columns
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    final = pd.concat(
        [
            week7_valid,
            week8_valid
        ],
        ignore_index=True
    )

    final = final.sort_values(
        "recording_id"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    total_rows = len(
        final
    )

    unique_recording = (
        final["recording_id"]
        .nunique()
    )

    unique_source = (
        final["source_file_id"]
        .nunique()
    )

    duplicate_recording = int(
        final["recording_id"]
        .duplicated()
        .sum()
    )

    # Recording per source
    per_source = (
        final
        .groupby("source_file_id")
        .size()
    )

    bad_source_count = (
        per_source[
            per_source != 4
        ]
    )

    # --------------------------------------------------------
    # CHECK RECORDING STATUS
    # --------------------------------------------------------

    done_count = int(
        (
            clean_status(
                final["recording_status"]
            )
            == "done"
        ).sum()
    )

    pass_count = int(
        (
            final["qc_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            == "PASS"
        ).sum()
    )

    # --------------------------------------------------------
    # CHECK QC
    # --------------------------------------------------------

    clip_ratio = pd.to_numeric(
        final["qc_clip_ratio"],
        errors="coerce"
    )

    clip_count = int(
        (
            clip_ratio > 0
        ).sum()
    )

    finite_values = (
        final["qc_finite"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    finite_false = int(
        (
            finite_values == "false"
        ).sum()
    )

    # --------------------------------------------------------
    # ASSERT FINAL DESIGN
    # --------------------------------------------------------

    if total_rows != 480:
        raise ValueError(
            f"Final manifest harus 480 row, "
            f"tetapi ditemukan {total_rows}."
        )

    if unique_recording != 480:
        raise ValueError(
            f"Harus ada 480 recording ID unik, "
            f"tetapi ditemukan {unique_recording}."
        )

    if duplicate_recording != 0:
        raise ValueError(
            f"Ditemukan {duplicate_recording} "
            f"duplicate recording ID."
        )

    if unique_source != 120:
        raise ValueError(
            f"Harus ada 120 source unik, "
            f"tetapi ditemukan {unique_source}."
        )

    if len(bad_source_count) != 0:
        print()
        print(
            "Source dengan jumlah recording "
            "tidak sama dengan 4:"
        )
        print(
            bad_source_count
        )

        raise ValueError(
            "Setiap source harus memiliki "
            "tepat 4 recording."
        )

    if done_count != 480:
        raise ValueError(
            f"DONE harus 480, ditemukan "
            f"{done_count}."
        )

    if pass_count != 480:
        raise ValueError(
            f"PASS harus 480, ditemukan "
            f"{pass_count}."
        )

    if clip_count != 0:
        raise ValueError(
            f"Ditemukan {clip_count} recording "
            f"dengan qc_clip_ratio > 0."
        )

    if finite_false != 0:
        raise ValueError(
            f"Ditemukan {finite_false} recording "
            f"dengan qc_finite = False."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("==============================================")
    print("FINAL MANIFEST 480 BERHASIL DIBUAT")
    print("==============================================")

    print(
        "Output             :",
        OUTPUT_MANIFEST
    )

    print(
        "Rows               :",
        total_rows
    )

    print(
        "Unique recording   :",
        unique_recording
    )

    print(
        "Unique source      :",
        unique_source
    )

    print(
        "DONE               :",
        done_count
    )

    print(
        "PASS               :",
        pass_count
    )

    print(
        "Duplicate IDs      :",
        duplicate_recording
    )

    print(
        "Clip ratio > 0     :",
        clip_count
    )

    print(
        "Finite False       :",
        finite_false
    )

    print()

    print("Repetition:")
    print(
        final[
            "repetition"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print("Label:")
    print(
        final[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    if "room" in final.columns:
        print()
        print("Room:")
        print(
            final[
                "room"
            ]
            .value_counts()
            .sort_index()
        )

    if "mic" in final.columns:
        print()
        print("Mic:")
        print(
            final[
                "mic"
            ]
            .value_counts()
        )

    if "playback" in final.columns:
        print()
        print("Playback:")
        print(
            final[
                "playback"
            ]
            .value_counts()
        )

    print()

    print(
        "Recording per source:"
    )

    print(
        per_source
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "MINGGU 8 SELESAI:"
    )

    print(
        "480 controlled real replay "
        "telah memiliki manifest final."
    )

if __name__ == "__main__":
    main()