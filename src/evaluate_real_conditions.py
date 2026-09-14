from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score
)


# ============================================================
# CONFIGURATION
# ============================================================

REAL_SCORES = Path(
    "results/scores_real_replay.csv"
)

REAL_MANIFEST = Path(
    "manifests/real_replay_manifest_480.csv"
)

TRAIN_MANIFEST = Path(
    "manifests/train_replay_manifest.csv"
)

RESULT_DIR = Path(
    "results"
)


# ============================================================
# METRICS
# ============================================================

def compute_eer(
    y_true,
    scores
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
        pos_label=1
    )

    fnr = (
        1.0 - tpr
    )

    index = np.nanargmin(
        np.abs(
            fpr - fnr
        )
    )

    eer = float(
        (
            fpr[index]
            +
            fnr[index]
        )
        /
        2.0
    )

    eer_threshold = float(
        thresholds[index]
    )

    return (
        eer,
        eer_threshold
    )


def compute_metrics(
    df
):

    if len(df) == 0:

        return None

    if df[
        "label"
    ].nunique() < 2:

        return None

    y_true = (
        df[
            "label"
        ]
        .astype(int)
        .to_numpy()
    )

    scores = (
        df[
            "fake_score"
        ]
        .astype(float)
        .to_numpy()
    )

    threshold_values = (
        df[
            "fixed_threshold"
        ]
        .astype(float)
        .unique()
    )

    if len(
        threshold_values
    ) != 1:

        raise RuntimeError(
            "Subset memiliki lebih dari satu "
            "fixed threshold."
        )

    fixed_threshold = float(
        threshold_values[0]
    )

    eer, eer_threshold = (
        compute_eer(
            y_true,
            scores
        )
    )

    auc = float(
        roc_auc_score(
            y_true,
            scores
        )
    )

    prediction = (
        scores
        >=
        fixed_threshold
    ).astype(int)

    accuracy = float(
        (
            prediction
            ==
            y_true
        ).mean()
    )

    f1 = float(
        f1_score(
            y_true,
            prediction,
            zero_division=0
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            y_true,
            prediction
        )
    )

    return {

        "n":
            len(
                df
            ),

        "label_0":
            int(
                (
                    y_true == 0
                ).sum()
            ),

        "label_1":
            int(
                (
                    y_true == 1
                ).sum()
            ),

        "eer":
            eer,

        "eer_threshold":
            eer_threshold,

        "auc":
            auc,

        "f1":
            f1,

        "balanced_accuracy":
            balanced_accuracy,

        "accuracy":
            accuracy,

        "fixed_threshold":
            fixed_threshold
    }


# ============================================================
# ADD RESULT ROW
# ============================================================

def add_result(
    rows,
    model_name,
    condition_type,
    condition_value,
    subset
):

    metrics = compute_metrics(
        subset
    )

    if metrics is None:
        return

    row = {

        "model":
            model_name,

        "condition_type":
            condition_type,

        "condition_value":
            str(
                condition_value
            )
    }

    row.update(
        metrics
    )

    rows.append(
        row
    )


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_condition_comparison(
    result_df,
    condition_type
):

    sub = result_df[
        result_df[
            "condition_type"
        ]
        ==
        condition_type
    ]

    if len(sub) == 0:
        return

    print()
    print(
        "=============================================="
    )

    print(
        condition_type.upper()
    )

    print(
        "=============================================="
    )

    values = sorted(
        sub[
            "condition_value"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    for value in values:

        print()
        print(
            f"Condition: {value}"
        )

        condition_df = sub[
            sub[
                "condition_value"
            ]
            ==
            value
        ]

        for model_name in [
            "no_grl",
            "with_grl"
        ]:

            model_df = condition_df[
                condition_df[
                    "model"
                ]
                ==
                model_name
            ]

            if len(model_df) == 0:
                continue

            row = model_df.iloc[
                0
            ]

            print(
                f"{model_name:10s} "
                f"| n={int(row['n']):3d} "
                f"| EER={row['eer'] * 100:6.2f}% "
                f"| AUC={row['auc']:.4f} "
                f"| F1={row['f1']:.4f} "
                f"| BAcc={row['balanced_accuracy']:.4f}"
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
        "TAHAP 8 - REAL REPLAY CONDITION ANALYSIS"
    )

    print(
        "=============================================="
    )

    # ========================================================
    # CHECK FILES
    # ========================================================

    required_files = [

        REAL_SCORES,
        REAL_MANIFEST,
        TRAIN_MANIFEST
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Tidak ditemukan: {path}"
            )

    # ========================================================
    # LOAD SCORES
    # ========================================================

    scores = pd.read_csv(
        REAL_SCORES
    )

    required_score_columns = [

        "model",
        "recording_id",
        "source_file_id",
        "label",
        "fake_score",
        "fixed_threshold",
        "room",
        "mic",
        "playback",
        "distance",
        "angle",
        "repetition"
    ]

    missing = [

        column

        for column in required_score_columns

        if column not in scores.columns
    ]

    if missing:

        raise ValueError(
            "scores_real_replay.csv kehilangan "
            f"kolom: {missing}"
        )

    scores[
        "label"
    ] = pd.to_numeric(
        scores[
            "label"
        ],
        errors="raise"
    ).astype(int)

    # ========================================================
    # EXPECTED SCORE SIZE
    # ========================================================

    print()
    print(
        "Score rows:",
        len(
            scores
        )
    )

    print(
        "Models:"
    )

    print(
        scores[
            "model"
        ]
        .value_counts()
    )

    expected_models = {
        "no_grl",
        "with_grl"
    }

    actual_models = set(
        scores[
            "model"
        ]
        .astype(str)
        .unique()
    )

    if actual_models != expected_models:

        raise RuntimeError(
            "Model pada score file tidak sesuai. "
            f"Ditemukan: {actual_models}"
        )

    for model_name in expected_models:

        count = int(
            (
                scores[
                    "model"
                ]
                ==
                model_name
            ).sum()
        )

        if count != 480:

            raise RuntimeError(
                f"{model_name} harus memiliki "
                f"480 scores, ditemukan {count}."
            )

    # ========================================================
    # LOAD MANIFESTS
    # ========================================================

    real_manifest = pd.read_csv(
        REAL_MANIFEST
    )

    train_manifest = pd.read_csv(
        TRAIN_MANIFEST
    )

    # ========================================================
    # IDENTIFY SOURCE OVERLAP
    # ========================================================

    real_sources = set(
        real_manifest[
            "source_file_id"
        ]
        .astype(str)
    )

    train_sources = set(
        train_manifest[
            "source_file_id"
        ]
        .astype(str)
    )

    overlap_sources = sorted(
        real_sources
        &
        train_sources
    )

    print()
    print(
        "Real unique source:",
        len(
            real_sources
        )
    )

    print(
        "Source overlap dengan training:",
        len(
            overlap_sources
        )
    )

    if overlap_sources:

        print(
            "Overlap source ID:"
        )

        for source_id in overlap_sources:

            print(
                " ",
                source_id
            )

    # ========================================================
    # SOURCE-DISJOINT SCORE SET
    # ========================================================

    if overlap_sources:

        source_disjoint_scores = scores[
            ~scores[
                "source_file_id"
            ]
            .astype(str)
            .isin(
                overlap_sources
            )
        ].copy()

    else:

        source_disjoint_scores = (
            scores.copy()
        )

    print()
    print(
        "Rows overall:",
        len(
            scores
        )
    )

    print(
        "Rows source-disjoint:",
        len(
            source_disjoint_scores
        )
    )

    expected_disjoint = (
        476
        if len(
            overlap_sources
        ) == 1
        else None
    )

    if (
        expected_disjoint is not None
        and
        len(
            source_disjoint_scores
        ) !=
        expected_disjoint * 2
    ):

        print(
            "PERINGATAN:"
        )

        print(
            "Karena score file berisi dua model, "
            "source-disjoint seharusnya "
            f"{expected_disjoint * 2} rows."
        )

    # ========================================================
    # ANALYSIS
    # ========================================================

    result_rows = []

    # ========================================================
    # OVERALL 480
    # ========================================================

    for model_name in [
        "no_grl",
        "with_grl"
    ]:

        model_df = scores[
            scores[
                "model"
            ]
            ==
            model_name
        ]

        add_result(
            result_rows,
            model_name,
            "overall",
            "real_replay_480",
            model_df
        )

    # ========================================================
    # SOURCE-DISJOINT 476
    # ========================================================

    for model_name in [
        "no_grl",
        "with_grl"
    ]:

        model_df = source_disjoint_scores[
            source_disjoint_scores[
                "model"
            ]
            ==
            model_name
        ]

        add_result(
            result_rows,
            model_name,
            "source_disjoint",
            "real_replay_without_train_overlap",
            model_df
        )

    # ========================================================
    # CONDITION ANALYSIS
    # ========================================================

    condition_columns = [

        "room",
        "mic",
        "playback",
        "distance",
        "angle",
        "repetition"
    ]

    for condition_column in condition_columns:

        print()
        print(
            f"Processing condition: "
            f"{condition_column}"
        )

        unique_values = (
            scores[
                condition_column
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        for condition_value in unique_values:

            for model_name in [
                "no_grl",
                "with_grl"
            ]:

                subset = scores[

                    (
                        scores[
                            "model"
                        ]
                        ==
                        model_name
                    )

                    &

                    (
                        scores[
                            condition_column
                        ]
                        .astype(str)
                        ==
                        str(
                            condition_value
                        )
                    )

                ]

                add_result(
                    result_rows,
                    model_name,
                    condition_column,
                    condition_value,
                    subset
                )

    # ========================================================
    # EXPLICIT MIC GROUP
    # ========================================================
    #
    # mic_A + mic_B -> matched/seen-like physical microphones
    # mic_unseen    -> explicit unseen microphone
    #
    # Ini khusus sesuai acquisition design Anda.
    # ========================================================

    for model_name in [
        "no_grl",
        "with_grl"
    ]:

        model_df = scores[
            scores[
                "model"
            ]
            ==
            model_name
        ]

        matched_mic = model_df[
            model_df[
                "mic"
            ]
            .astype(str)
            .isin(
                [
                    "mic_A",
                    "mic_B"
                ]
            )
        ]

        unseen_mic = model_df[
            model_df[
                "mic"
            ]
            .astype(str)
            ==
            "mic_unseen"
        ]

        add_result(
            result_rows,
            model_name,
            "mic_group",
            "matched_mic_A_B",
            matched_mic
        )

        add_result(
            result_rows,
            model_name,
            "mic_group",
            "mic_unseen",
            unseen_mic
        )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        result_rows
    )

    if len(
        result_df
    ) == 0:

        raise RuntimeError(
            "Tidak ada result condition."
        )

    # ========================================================
    # GRL DELTA TABLE
    # ========================================================

    delta_rows = []

    grouped_keys = (
        result_df[
            [
                "condition_type",
                "condition_value"
            ]
        ]
        .drop_duplicates()
    )

    for _, key_row in grouped_keys.iterrows():

        condition_type = (
            key_row[
                "condition_type"
            ]
        )

        condition_value = (
            key_row[
                "condition_value"
            ]
        )

        condition_data = result_df[

            (
                result_df[
                    "condition_type"
                ]
                ==
                condition_type
            )

            &

            (
                result_df[
                    "condition_value"
                ]
                ==
                condition_value
            )

        ]

        no_grl = condition_data[
            condition_data[
                "model"
            ]
            ==
            "no_grl"
        ]

        with_grl = condition_data[
            condition_data[
                "model"
            ]
            ==
            "with_grl"
        ]

        if (
            len(
                no_grl
            ) != 1
            or
            len(
                with_grl
            ) != 1
        ):

            continue

        no_row = no_grl.iloc[
            0
        ]

        grl_row = with_grl.iloc[
            0
        ]

        delta_rows.append({

            "condition_type":
                condition_type,

            "condition_value":
                condition_value,

            "n":
                int(
                    no_row[
                        "n"
                    ]
                ),

            "no_grl_eer":
                no_row[
                    "eer"
                ],

            "with_grl_eer":
                grl_row[
                    "eer"
                ],

            "eer_improvement_pp":
                (
                    no_row[
                        "eer"
                    ]
                    -
                    grl_row[
                        "eer"
                    ]
                )
                *
                100.0,

            "no_grl_auc":
                no_row[
                    "auc"
                ],

            "with_grl_auc":
                grl_row[
                    "auc"
                ],

            "auc_difference":
                (
                    grl_row[
                        "auc"
                    ]
                    -
                    no_row[
                        "auc"
                    ]
                ),

            "no_grl_bacc":
                no_row[
                    "balanced_accuracy"
                ],

            "with_grl_bacc":
                grl_row[
                    "balanced_accuracy"
                ],

            "bacc_difference":
                (
                    grl_row[
                        "balanced_accuracy"
                    ]
                    -
                    no_row[
                        "balanced_accuracy"
                    ]
                )
        })

    delta_df = pd.DataFrame(
        delta_rows
    )

    # ========================================================
    # SAVE
    # ========================================================

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    conditions_path = (
        RESULT_DIR
        /
        "real_replay_condition_analysis.csv"
    )

    delta_path = (
        RESULT_DIR
        /
        "real_replay_condition_grl_comparison.csv"
    )

    disjoint_path = (
        RESULT_DIR
        /
        "scores_real_replay_source_disjoint.csv"
    )

    config_path = (
        RESULT_DIR
        /
        "real_replay_condition_analysis_config.json"
    )

    result_df.to_csv(
        conditions_path,
        index=False
    )

    delta_df.to_csv(
        delta_path,
        index=False
    )

    source_disjoint_scores.to_csv(
        disjoint_path,
        index=False
    )

    config = {

        "analysis":
            "controlled_real_replay_conditions",

        "score_file":
            str(
                REAL_SCORES
            ),

        "original_recordings":
            480,

        "models": [
            "no_grl",
            "with_grl"
        ],

        "condition_columns": [
            "room",
            "mic",
            "playback",
            "distance",
            "angle",
            "repetition"
        ],

        "explicit_unseen_factor":
            "mic_unseen",

        "matched_mic_group": [
            "mic_A",
            "mic_B"
        ],

        "train_overlap_sources":
            overlap_sources,

        "n_train_overlap_sources":
            len(
                overlap_sources
            )
    }

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "REAL REPLAY CONDITION ANALYSIS SELESAI"
    )

    print(
        "=============================================="
    )

    print_condition_comparison(
        result_df,
        "overall"
    )

    print_condition_comparison(
        result_df,
        "source_disjoint"
    )

    print_condition_comparison(
        result_df,
        "mic_group"
    )

    print_condition_comparison(
        result_df,
        "mic"
    )

    print_condition_comparison(
        result_df,
        "room"
    )

    print_condition_comparison(
        result_df,
        "playback"
    )

    print_condition_comparison(
        result_df,
        "distance"
    )

    print_condition_comparison(
        result_df,
        "angle"
    )

    print_condition_comparison(
        result_df,
        "repetition"
    )

    # ========================================================
    # EXPLICIT UNSEEN MIC SUMMARY
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "UNSEEN MICROPHONE SUMMARY"
    )

    print(
        "=============================================="
    )

    unseen_rows = result_df[

        (
            result_df[
                "condition_type"
            ]
            ==
            "mic_group"
        )

        &

        (
            result_df[
                "condition_value"
            ]
            ==
            "mic_unseen"
        )

    ]

    matched_rows = result_df[

        (
            result_df[
                "condition_type"
            ]
            ==
            "mic_group"
        )

        &

        (
            result_df[
                "condition_value"
            ]
            ==
            "matched_mic_A_B"
        )

    ]

    for model_name in [
        "no_grl",
        "with_grl"
    ]:

        unseen_model = unseen_rows[
            unseen_rows[
                "model"
            ]
            ==
            model_name
        ]

        matched_model = matched_rows[
            matched_rows[
                "model"
            ]
            ==
            model_name
        ]

        if (
            len(
                unseen_model
            ) == 1
            and
            len(
                matched_model
            ) == 1
        ):

            unseen_row = (
                unseen_model
                .iloc[
                    0
                ]
            )

            matched_row = (
                matched_model
                .iloc[
                    0
                ]
            )

            gap = (
                unseen_row[
                    "eer"
                ]
                -
                matched_row[
                    "eer"
                ]
            )

            print()
            print(
                model_name.upper()
            )

            print(
                "Matched mic EER:",
                f"{matched_row['eer'] * 100:.2f}%"
            )

            print(
                "Unseen mic EER :",
                f"{unseen_row['eer'] * 100:.2f}%"
            )

            print(
                "Unseen mic gap :",
                f"{gap * 100:+.2f}",
                "percentage points"
            )

    # ========================================================
    # SOURCE-DISJOINT SENSITIVITY
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "SOURCE-DISJOINT SENSITIVITY"
    )

    print(
        "=============================================="
    )

    for model_name in [
        "no_grl",
        "with_grl"
    ]:

        overall = result_df[

            (
                result_df[
                    "model"
                ]
                ==
                model_name
            )

            &

            (
                result_df[
                    "condition_type"
                ]
                ==
                "overall"
            )

        ].iloc[
            0
        ]

        disjoint = result_df[

            (
                result_df[
                    "model"
                ]
                ==
                model_name
            )

            &

            (
                result_df[
                    "condition_type"
                ]
                ==
                "source_disjoint"
            )

        ].iloc[
            0
        ]

        difference = (
            disjoint[
                "eer"
            ]
            -
            overall[
                "eer"
            ]
        )

        print()
        print(
            model_name.upper()
        )

        print(
            "Overall EER        :",
            f"{overall['eer'] * 100:.2f}%"
        )

        print(
            "Source-disjoint EER:",
            f"{disjoint['eer'] * 100:.2f}%"
        )

        print(
            "Difference         :",
            f"{difference * 100:+.2f}",
            "percentage points"
        )

    # ========================================================
    # FILE OUTPUT
    # ========================================================

    print()
    print(
        "Output:"
    )

    print(
        conditions_path
    )

    print(
        delta_path
    )

    print(
        disjoint_path
    )

    print(
        config_path
    )

if __name__ == "__main__":
    main()