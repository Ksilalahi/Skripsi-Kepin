from pathlib import Path

import pandas as pd


# ============================================================
# PATH
# ============================================================

INPUT_MANIFEST = Path(
    "manifests/train_replay_manifest.csv"
)

OUTPUT_MANIFEST = Path(
    "manifests/train_e2_clean_manifest.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=============================================="
    )
    print(
        "E2 - MEMBUAT CLEAN-ONLY TRAIN MANIFEST"
    )
    print(
        "=============================================="
    )

    if not INPUT_MANIFEST.exists():

        raise FileNotFoundError(
            INPUT_MANIFEST
        )

    df = pd.read_csv(
        INPUT_MANIFEST
    )

    required_columns = [
        "file_id",
        "source_file_id",
        "file_path",
        "speaker_id",
        "attack_id",
        "label",
        "dataset",
        "split",
        "condition",
        "transform_label",
        "domain_label"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Manifest kehilangan kolom: "
            f"{missing}"
        )

    print()
    print(
        "Input rows:",
        len(df)
    )

    # ========================================================
    # CLEAN ONLY
    # ========================================================

    clean = df[
        df[
            "condition"
        ]
        ==
        "clean"
    ].copy()

    print(
        "Clean rows:",
        len(clean)
    )

    # ========================================================
    # TRAIN / VALIDATION ONLY
    # ========================================================

    clean = clean[
        clean[
            "split"
        ].isin(
            [
                "train",
                "validation"
            ]
        )
    ].copy()

    # ========================================================
    # EXPECTED
    # ========================================================

    train_df = clean[
        clean[
            "split"
        ]
        ==
        "train"
    ]

    validation_df = clean[
        clean[
            "split"
        ]
        ==
        "validation"
    ]

    print()
    print(
        "Train clean:",
        len(train_df)
    )

    print(
        "Validation clean:",
        len(validation_df)
    )

    if len(
        train_df
    ) != 600:

        raise RuntimeError(
            "E2 train clean seharusnya 600, "
            f"ditemukan {len(train_df)}."
        )

    if len(
        validation_df
    ) != 200:

        raise RuntimeError(
            "E2 validation clean seharusnya 200, "
            f"ditemukan {len(validation_df)}."
        )

    # ========================================================
    # LABEL BALANCE
    # ========================================================

    print()
    print(
        "Train label:"
    )

    print(
        train_df[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Validation label:"
    )

    print(
        validation_df[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    expected_train = {
        0: 300,
        1: 300
    }

    expected_validation = {
        0: 100,
        1: 100
    }

    for label, expected in \
            expected_train.items():

        actual = int(
            (
                train_df[
                    "label"
                ]
                ==
                label
            ).sum()
        )

        if actual != expected:

            raise RuntimeError(
                f"Train label {label}: "
                f"expected={expected}, "
                f"actual={actual}"
            )

    for label, expected in \
            expected_validation.items():

        actual = int(
            (
                validation_df[
                    "label"
                ]
                ==
                label
            ).sum()
        )

        if actual != expected:

            raise RuntimeError(
                f"Validation label {label}: "
                f"expected={expected}, "
                f"actual={actual}"
            )

    # ========================================================
    # TRANSFORM / DOMAIN
    # ========================================================

    clean[
        "transform_label"
    ] = 0

    clean[
        "domain_label"
    ] = -1

    clean[
        "domain_name"
    ] = "unknown"

    # ========================================================
    # SOURCE LEAKAGE
    # ========================================================

    train_sources = set(
        train_df[
            "source_file_id"
        ]
        .astype(str)
    )

    validation_sources = set(
        validation_df[
            "source_file_id"
        ]
        .astype(str)
    )

    overlap = (
        train_sources
        &
        validation_sources
    )

    print()
    print(
        "Source leakage:",
        len(overlap)
    )

    if overlap:

        raise RuntimeError(
            "Source leakage ditemukan."
        )

    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    duplicate_file_id = int(
        clean[
            "file_id"
        ]
        .duplicated()
        .sum()
    )

    duplicate_path = int(
        clean[
            "file_path"
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate file_id:",
        duplicate_file_id
    )

    print(
        "Duplicate file_path:",
        duplicate_path
    )

    if duplicate_file_id > 0:

        raise RuntimeError(
            "Duplicate file_id ditemukan."
        )

    if duplicate_path > 0:

        raise RuntimeError(
            "Duplicate file_path ditemukan."
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    clean.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "E2 CLEAN-ONLY MANIFEST BERHASIL DIBUAT"
    )

    print(
        "=============================================="
    )

    print(
        "Output:",
        OUTPUT_MANIFEST
    )

    print(
        "Rows:",
        len(clean)
    )

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Validation:",
        len(validation_df)
    )

    print(
        "Source leakage:",
        len(overlap)
    )

if __name__ == "__main__":
    main()