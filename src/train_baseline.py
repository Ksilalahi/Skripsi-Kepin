from pathlib import Path

import pandas as pd


# ============================================================
# PATH
# ============================================================

SOURCE_MANIFEST = Path(
    "manifests/source_manifest.csv"
)

SIM_MANIFEST = Path(
    "manifests/manifest_replay_sim_week9.csv"
)

OUTPUT_MANIFEST = Path(
    "manifests/train_replay_manifest.csv"
)

DOMAIN_MAPPING_PATH = Path(
    "manifests/domain_mapping.csv"
)


# ============================================================
# HELPER
# ============================================================

def normalize_source_split(value):
    """
    Normalisasi nama split source manifest.

    train -> train
    dev   -> validation
    eval  -> test
    """

    value = str(value).strip().lower()

    if value == "train":
        return "train"

    if value in [
        "dev",
        "validation",
        "val"
    ]:
        return "validation"

    if value in [
        "eval",
        "test"
    ]:
        return "test"

    return value


def clean_string(series):
    """
    Membersihkan kolom string.
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
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
        "MEMBUAT TRAIN REPLAY MANIFEST - WEEK 9"
    )
    print(
        "=============================================="
    )
    print()

    # ========================================================
    # CHECK INPUT FILES
    # ========================================================

    if not SOURCE_MANIFEST.exists():

        raise FileNotFoundError(
            f"Tidak ditemukan: "
            f"{SOURCE_MANIFEST}"
        )

    if not SIM_MANIFEST.exists():

        raise FileNotFoundError(
            f"Tidak ditemukan: "
            f"{SIM_MANIFEST}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    source = pd.read_csv(
        SOURCE_MANIFEST
    )

    sim = pd.read_csv(
        SIM_MANIFEST
    )

    print(
        "Source manifest rows :",
        len(source)
    )

    print(
        "Sim Week 9 rows      :",
        len(sim)
    )

    # ========================================================
    # VALIDATE SOURCE MANIFEST
    # ========================================================

    source_required = [
        "source_file_id",
        "file_path",
        "speaker_id",
        "attack_id",
        "label",
        "dataset",
        "split"
    ]

    missing_source = [
        column
        for column in source_required
        if column not in source.columns
    ]

    if missing_source:

        raise ValueError(
            "source_manifest.csv kehilangan "
            f"kolom: {missing_source}"
        )

    # ========================================================
    # VALIDATE SIM MANIFEST
    # ========================================================

    sim_required = [
        "file_id",
        "source_file_id",
        "file_path",
        "speaker_id",
        "attack_id",
        "label",
        "dataset",
        "source_split",
        "split",
        "condition",
        "rir_identity",
        "rir_group",
        "snr_db",
        "seed",
        "qc_status"
    ]

    missing_sim = [
        column
        for column in sim_required
        if column not in sim.columns
    ]

    if missing_sim:

        raise ValueError(
            "manifest_replay_sim_week9.csv "
            f"kehilangan kolom: {missing_sim}"
        )

    # ========================================================
    # DATA TYPES
    # ========================================================

    source["label"] = pd.to_numeric(
        source["label"],
        errors="raise"
    ).astype(int)

    sim["label"] = pd.to_numeric(
        sim["label"],
        errors="raise"
    ).astype(int)

    source["normalized_split"] = (
        source["split"]
        .apply(
            normalize_source_split
        )
    )

    sim["split"] = (
        clean_string(
            sim["split"]
        )
        .str.lower()
    )

    sim["rir_group"] = (
        clean_string(
            sim["rir_group"]
        )
        .str.lower()
    )

    sim["qc_status"] = (
        clean_string(
            sim["qc_status"]
        )
        .str.upper()
    )

    # ========================================================
    # FILTER SIMULATED REPLAY
    # ========================================================
    #
    # Hanya:
    #
    # train + validation
    # RIR seen
    # QC PASS
    #
    # TEST UNSEEN TIDAK MASUK TRAINING.
    # ========================================================

    sim_trainval = sim[
        sim["split"].isin(
            [
                "train",
                "validation"
            ]
        )
        &
        (
            sim["rir_group"]
            == "seen"
        )
        &
        (
            sim["qc_status"]
            == "PASS"
        )
    ].copy()

    print()
    print(
        "Sim train+validation seen:",
        len(sim_trainval)
    )

    if len(sim_trainval) == 0:

        raise RuntimeError(
            "Tidak ada simulated replay "
            "train/validation seen."
        )

    # ========================================================
    # EXPECTED SIMULATED DISTRIBUTION
    # ========================================================

    sim_distribution = pd.crosstab(
        sim_trainval["split"],
        sim_trainval["label"]
    )

    print()
    print(
        "Distribusi simulated:"
    )
    print(
        sim_distribution
    )

    # Target dari dataset Week 9
    expected_sim = {
        "train": {
            0: 300,
            1: 300
        },
        "validation": {
            0: 100,
            1: 100
        }
    }

    for split_name, label_counts \
            in expected_sim.items():

        part = sim_trainval[
            sim_trainval["split"]
            == split_name
        ]

        counts = (
            part["label"]
            .value_counts()
        )

        for label, expected in \
                label_counts.items():

            actual = int(
                counts.get(
                    label,
                    0
                )
            )

            if actual != expected:

                raise RuntimeError(
                    f"Simulated {split_name}, "
                    f"label {label}: "
                    f"expected={expected}, "
                    f"actual={actual}"
                )

    # ========================================================
    # GET SOURCE IDS USED BY SIMULATED REPLAY
    # ========================================================
    #
    # Clean counterpart diambil dari source yang SAMA.
    #
    # Ini menjaga pairing:
    #
    # source X clean
    # source X simulated
    #
    # dan tetap berada pada split yang sama.
    # ========================================================

    selected_source_ids = set(
        sim_trainval[
            "source_file_id"
        ]
        .astype(str)
    )

    clean = source[
        source[
            "source_file_id"
        ]
        .astype(str)
        .isin(
            selected_source_ids
        )
    ].copy()

    # Hanya train + validation
    clean = clean[
        clean[
            "normalized_split"
        ].isin(
            [
                "train",
                "validation"
            ]
        )
    ].copy()

    clean["split"] = (
        clean["normalized_split"]
    )

    # ========================================================
    # VERIFY CLEAN COUNTERPART
    # ========================================================

    sim_source_count = (
        sim_trainval[
            "source_file_id"
        ]
        .nunique()
    )

    clean_source_count = (
        clean[
            "source_file_id"
        ]
        .nunique()
    )

    print()
    print(
        "Sim source unique   :",
        sim_source_count
    )

    print(
        "Clean counterpart   :",
        clean_source_count
    )

    if (
        clean_source_count
        !=
        sim_source_count
    ):

        sim_ids = set(
            sim_trainval[
                "source_file_id"
            ]
            .astype(str)
        )

        clean_ids = set(
            clean[
                "source_file_id"
            ]
            .astype(str)
        )

        missing_ids = (
            sim_ids
            -
            clean_ids
        )

        print()
        print(
            "Source tidak ditemukan "
            "di clean manifest:"
        )

        print(
            list(
                missing_ids
            )[:20]
        )

        raise RuntimeError(
            "Tidak semua simulated replay "
            "memiliki clean counterpart."
        )

    # ========================================================
    # CHECK DUPLICATE CLEAN SOURCE
    # ========================================================

    duplicate_clean = int(
        clean[
            "source_file_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_clean > 0:

        raise RuntimeError(
            f"Clean source duplicate: "
            f"{duplicate_clean}"
        )

    # ========================================================
    # CLEAN METADATA
    # ========================================================

    clean["file_id"] = (
        clean["source_file_id"]
        .astype(str)
    )

    clean["condition"] = (
        "clean"
    )

    clean["transform_label"] = 0

    # Clean tidak punya RIR domain.
    # -1 berarti domain unknown dan
    # domain loss akan di-mask.
    clean["domain_label"] = -1

    clean["domain_name"] = (
        "unknown"
    )

    clean["rir_identity"] = ""
    clean["rir_group"] = ""
    clean["snr_db"] = ""
    clean["seed"] = ""

    # ========================================================
    # DOMAIN MAPPING FOR SIMULATED REPLAY
    # ========================================================
    #
    # 5 RIR seen identities:
    #
    # booth
    # lecture
    # meeting
    # office
    # phone
    #
    # Domain label menjadi:
    #
    # 0 ... 4
    # ========================================================

    domain_names = sorted(
        sim_trainval[
            "rir_identity"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(domain_names) != 5:

        print()
        print(
            "Domain ditemukan:"
        )

        print(
            domain_names
        )

        raise RuntimeError(
            "Seharusnya ada tepat "
            "5 RIR seen domains."
        )

    domain_mapping = {
        domain_name: index
        for index, domain_name
        in enumerate(
            domain_names
        )
    }

    sim_trainval[
        "domain_name"
    ] = (
        sim_trainval[
            "rir_identity"
        ]
        .astype(str)
    )

    sim_trainval[
        "domain_label"
    ] = (
        sim_trainval[
            "domain_name"
        ]
        .map(
            domain_mapping
        )
    )

    if sim_trainval[
        "domain_label"
    ].isna().any():

        raise RuntimeError(
            "Ada simulated replay "
            "tanpa domain label."
        )

    sim_trainval[
        "domain_label"
    ] = (
        sim_trainval[
            "domain_label"
        ]
        .astype(int)
    )

    sim_trainval[
        "transform_label"
    ] = 1

    sim_trainval[
        "condition"
    ] = "simulated_replay"

    # ========================================================
    # COMMON FINAL COLUMNS
    # ========================================================

    final_columns = [
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
        "domain_label",
        "domain_name",
        "rir_identity",
        "rir_group",
        "snr_db",
        "seed"
    ]

    clean_final = clean[
        final_columns
    ].copy()

    sim_final = sim_trainval[
        final_columns
    ].copy()

    # ========================================================
    # MERGE CLEAN + SIMULATED
    # ========================================================

    final = pd.concat(
        [
            clean_final,
            sim_final
        ],
        ignore_index=True
    )

    # ========================================================
    # GENERAL VALIDATION
    # ========================================================

    if final[
        "file_path"
    ].isna().any():

        raise RuntimeError(
            "Ada file_path kosong."
        )

    if final[
        "source_file_id"
    ].isna().any():

        raise RuntimeError(
            "Ada source_file_id kosong."
        )

    if not final[
        "label"
    ].isin(
        [0, 1]
    ).all():

        raise RuntimeError(
            "Label harus 0/1."
        )

    if not final[
        "transform_label"
    ].isin(
        [0, 1]
    ).all():

        raise RuntimeError(
            "transform_label "
            "harus 0/1."
        )

    if not (
        final[
            "domain_label"
        ] >= -1
    ).all():

        raise RuntimeError(
            "domain_label invalid."
        )

    # ========================================================
    # CRITICAL SOURCE LEAKAGE CHECK
    # ========================================================

    source_split_count = (
        final
        .groupby(
            "source_file_id"
        )[
            "split"
        ]
        .nunique()
    )

    leakage = (
        source_split_count[
            source_split_count > 1
        ]
    )

    if len(leakage) > 0:

        print()
        print(
            "SOURCE LEAKAGE:"
        )

        print(
            leakage.head(
                20
            )
        )

        raise RuntimeError(
            "Source leakage terdeteksi."
        )

    # ========================================================
    # CONDITION PAIR CHECK
    # ========================================================
    #
    # Setiap selected source harus mempunyai:
    #
    # 1 clean
    # 1 simulated replay
    #
    # total = 2 row/source
    # ========================================================

    rows_per_source = (
        final
        .groupby(
            "source_file_id"
        )
        .size()
    )

    bad_pairs = (
        rows_per_source[
            rows_per_source != 2
        ]
    )

    if len(bad_pairs) > 0:

        print()
        print(
            "Source tanpa pasangan "
            "clean + simulated:"
        )

        print(
            bad_pairs.head(
                20
            )
        )

        raise RuntimeError(
            "Pair clean/simulated "
            "tidak lengkap."
        )

    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    duplicate_file_id = int(
        final[
            "file_id"
        ]
        .duplicated()
        .sum()
    )

    duplicate_file_path = int(
        final[
            "file_path"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_file_id > 0:

        raise RuntimeError(
            f"Duplicate file_id: "
            f"{duplicate_file_id}"
        )

    if duplicate_file_path > 0:

        raise RuntimeError(
            f"Duplicate file_path: "
            f"{duplicate_file_path}"
        )

    # ========================================================
    # BALANCE CHECK
    # ========================================================

    # --------------------------------------------------------
    # SPLIT x CONDITION
    # --------------------------------------------------------

    split_condition = pd.crosstab(
        final["split"],
        final["condition"]
    )

    # Expected:
    #
    # train:
    # clean 600
    # simulated 600
    #
    # validation:
    # clean 200
    # simulated 200

    expected_condition = {
        "train": {
            "clean": 600,
            "simulated_replay": 600
        },
        "validation": {
            "clean": 200,
            "simulated_replay": 200
        }
    }

    for split_name, conditions \
            in expected_condition.items():

        part = final[
            final["split"]
            == split_name
        ]

        counts = (
            part["condition"]
            .value_counts()
        )

        for condition, expected \
                in conditions.items():

            actual = int(
                counts.get(
                    condition,
                    0
                )
            )

            if actual != expected:

                raise RuntimeError(
                    f"{split_name} "
                    f"{condition}: "
                    f"expected={expected}, "
                    f"actual={actual}"
                )

    # --------------------------------------------------------
    # SPLIT x LABEL
    # --------------------------------------------------------

    split_label = pd.crosstab(
        final["split"],
        final["label"]
    )

    expected_label = {
        "train": {
            0: 600,
            1: 600
        },
        "validation": {
            0: 200,
            1: 200
        }
    }

    for split_name, labels \
            in expected_label.items():

        part = final[
            final["split"]
            == split_name
        ]

        counts = (
            part["label"]
            .value_counts()
        )

        for label, expected \
                in labels.items():

            actual = int(
                counts.get(
                    label,
                    0
                )
            )

            if actual != expected:

                raise RuntimeError(
                    f"{split_name}, "
                    f"label={label}: "
                    f"expected={expected}, "
                    f"actual={actual}"
                )

    # ========================================================
    # SORT
    # ========================================================

    split_order = {
        "train": 0,
        "validation": 1
    }

    condition_order = {
        "clean": 0,
        "simulated_replay": 1
    }

    final["_split_order"] = (
        final["split"]
        .map(
            split_order
        )
    )

    final["_condition_order"] = (
        final["condition"]
        .map(
            condition_order
        )
    )

    final = (
        final
        .sort_values(
            [
                "_split_order",
                "source_file_id",
                "_condition_order"
            ]
        )
        .drop(
            columns=[
                "_split_order",
                "_condition_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SAVE DOMAIN MAPPING
    # ========================================================

    domain_mapping_df = pd.DataFrame(
        [
            {
                "domain_label":
                    label,

                "domain_name":
                    name
            }
            for name, label
            in domain_mapping.items()
        ]
    )

    DOMAIN_MAPPING_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    domain_mapping_df.to_csv(
        DOMAIN_MAPPING_PATH,
        index=False
    )

    # ========================================================
    # SAVE TRAINING MANIFEST
    # ========================================================

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        "TRAIN REPLAY MANIFEST BERHASIL DIBUAT"
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
        len(final)
    )

    print(
        "Unique source:",
        final[
            "source_file_id"
        ].nunique()
    )

    print(
        "Unique file:",
        final[
            "file_id"
        ].nunique()
    )

    print()

    print(
        "Split:"
    )

    print(
        final[
            "split"
        ]
        .value_counts()
    )

    print()

    print(
        "Condition:"
    )

    print(
        final[
            "condition"
        ]
        .value_counts()
    )

    print()

    print(
        "Split x Condition:"
    )

    print(
        split_condition
    )

    print()

    print(
        "Split x Label:"
    )

    print(
        split_label
    )

    print()

    print(
        "Transform label:"
    )

    print(
        final[
            "transform_label"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Domain label:"
    )

    print(
        final[
            "domain_label"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Domain mapping:"
    )

    print(
        domain_mapping_df
    )

    print()

    print(
        "Jumlah domain seen:",
        len(
            domain_mapping
        )
    )

    print()

    print(
        "Source leakage      :",
        len(
            leakage
        )
    )

    print(
        "Bad clean/sim pairs :",
        len(
            bad_pairs
        )
    )

    print(
        "Duplicate file_id   :",
        duplicate_file_id
    )

    print(
        "Duplicate file_path :",
        duplicate_file_path
    )

    print()

    print(
        "Rows/source:"
    )

    print(
        rows_per_source
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "SIMULATED TEST UNSEEN "
        "TIDAK DIMASUKKAN KE TRAINING."
    )

    print(
        "REAL REPLAY 480 "
        "TIDAK DIMASUKKAN KE TRAINING."
    )

    print()

    print(
        "HASIL:"
    )

    print(
        "Manifest training Week 9 "
        "siap untuk model tanpa GRL "
        "dan dengan GRL."
    )

if __name__ == "__main__":
    main()