from pathlib import Path
import json

import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn.functional as F

from scipy.signal import resample_poly
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score
)

from spectral_phase import (
    SpectralPhase,
    modified_group_delay
)

from model_replay import ReplayResNet


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SR = 16000
SECONDS = 4
TARGET_LENGTH = TARGET_SR * SECONDS

CHECKPOINT_PATH = Path(
    "checkpoints/replay_e2_full_clean.pt"
)

TEST_MANIFEST = Path(
    "manifests/e2_simulated_seen_test.csv"
)

OUTPUT_SCORES = Path(
    "results/scores_e2_simulated_seen_test.csv"
)

OUTPUT_SUMMARY = Path(
    "results/e2_simulated_seen_summary.csv"
)

OUTPUT_CONDITIONS = Path(
    "results/e2_simulated_seen_conditions.csv"
)

OUTPUT_CONFIG = Path(
    "results/e2_simulated_seen_config.json"
)

N_DOMAIN = 5
EMBEDDING_DIM = 256


# ============================================================
# AUDIO
# ============================================================

def load_audio_4s(
    path,
    target_sr=TARGET_SR,
    seconds=SECONDS
):

    path = Path(
        str(path)
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Audio tidak ditemukan: {path}"
        )

    audio, sr = sf.read(
        str(path),
        always_2d=True
    )

    # Multichannel -> mono
    audio = audio.mean(
        axis=1
    )

    # Resample
    if sr != target_sr:

        audio = resample_poly(
            audio,
            target_sr,
            sr
        )

    audio = audio.astype(
        np.float32
    )

    target_length = (
        target_sr
        *
        seconds
    )

    n = len(
        audio
    )

    # Pad
    if n < target_length:

        audio = np.pad(
            audio,
            (
                0,
                target_length - n
            ),
            mode="constant"
        )

    # Center crop
    elif n > target_length:

        start = (
            n
            -
            target_length
        ) // 2

        audio = audio[
            start:
            start + target_length
        ]

    # Sama dengan training loader
    peak = float(
        np.max(
            np.abs(
                audio
            )
        )
    )

    if peak > 1e-6:

        audio = (
            audio
            /
            peak
        )

    return (
        torch
        .from_numpy(
            audio
        )
        .float()
        .unsqueeze(0)
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    wav,
    feature_extractor,
    stats
):

    # wav [1,T] -> [1,1,T]
    wav = wav.unsqueeze(0)

    with torch.no_grad():

        logmel, phase = (
            feature_extractor(
                wav
            )
        )

        mgd = modified_group_delay(
            wav,
            feature_extractor.fb
        )

    # [1,80,397] -> [80,397]
    logmel = logmel.squeeze(0)
    phase = phase.squeeze(0)
    mgd = mgd.squeeze(0)

    # ========================================================
    # FINITE CHECK
    # ========================================================

    for name, x in [
        ("logmel", logmel),
        ("phase", phase),
        ("mgd", mgd)
    ]:

        if not torch.isfinite(
            x
        ).all():

            raise RuntimeError(
                f"{name} memiliki NaN/Inf."
            )

    # ========================================================
    # E2 CLEAN-TRAIN STATISTICS
    # ========================================================

    logmel = (
        logmel
        -
        float(
            stats[
                "logmel_mean"
            ]
        )
    ) / (
        float(
            stats[
                "logmel_std"
            ]
        )
        +
        1e-6
    )

    phase = (
        phase
        -
        float(
            stats[
                "phase_mean"
            ]
        )
    ) / (
        float(
            stats[
                "phase_std"
            ]
        )
        +
        1e-6
    )

    mgd = (
        mgd
        -
        float(
            stats[
                "mgd_mean"
            ]
        )
    ) / (
        float(
            stats[
                "mgd_std"
            ]
        )
        +
        1e-6
    )

    features = torch.stack(
        [
            logmel,
            phase,
            mgd
        ],
        dim=0
    )

    # [3,80,397]
    return features.float()


# ============================================================
# EER
# ============================================================

def calculate_eer(
    labels,
    scores
):

    labels = np.asarray(
        labels,
        dtype=int
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1
    )

    fnr = (
        1.0
        -
        tpr
    )

    index = int(
        np.nanargmin(
            np.abs(
                fpr
                -
                fnr
            )
        )
    )

    eer = float(
        (
            fpr[
                index
            ]
            +
            fnr[
                index
            ]
        )
        /
        2.0
    )

    threshold = float(
        thresholds[
            index
        ]
    )

    return (
        eer,
        threshold
    )


# ============================================================
# METRICS USING FROZEN CLEAN VALIDATION THRESHOLD
# ============================================================

def compute_metrics(
    labels,
    scores,
    fixed_threshold
):

    labels = np.asarray(
        labels,
        dtype=int
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(
        np.unique(
            labels
        )
    ) != 2:

        raise RuntimeError(
            "Metric membutuhkan label 0 dan 1."
        )

    eer, test_eer_threshold = (
        calculate_eer(
            labels,
            scores
        )
    )

    auc = float(
        roc_auc_score(
            labels,
            scores
        )
    )

    predictions = (
        scores
        >=
        fixed_threshold
    ).astype(int)

    f1 = float(
        f1_score(
            labels,
            predictions,
            zero_division=0
        )
    )

    bacc = float(
        balanced_accuracy_score(
            labels,
            predictions
        )
    )

    accuracy = float(
        accuracy_score(
            labels,
            predictions
        )
    )

    return {
        "eer": eer,
        "test_eer_threshold": test_eer_threshold,
        "auc": auc,
        "f1": f1,
        "balanced_accuracy": bacc,
        "accuracy": accuracy
    }


# ============================================================
# CONDITION METRICS
# ============================================================

def condition_metrics(
    scores_df,
    column,
    fixed_threshold
):

    rows = []

    for value, group in scores_df.groupby(
        column
    ):

        if group[
            "label"
        ].nunique() < 2:

            print(
                f"SKIP {column}={value}: "
                "hanya memiliki satu kelas."
            )

            continue

        metrics = compute_metrics(
            group[
                "label"
            ].values,
            group[
                "fake_score"
            ].values,
            fixed_threshold
        )

        rows.append({
            "condition_type":
                column,

            "condition":
                str(value),

            "n":
                len(group),

            "label_0":
                int(
                    (
                        group[
                            "label"
                        ]
                        ==
                        0
                    ).sum()
                ),

            "label_1":
                int(
                    (
                        group[
                            "label"
                        ]
                        ==
                        1
                    ).sum()
                ),

            "eer":
                metrics[
                    "eer"
                ],

            "auc":
                metrics[
                    "auc"
                ],

            "f1":
                metrics[
                    "f1"
                ],

            "balanced_accuracy":
                metrics[
                    "balanced_accuracy"
                ],

            "accuracy":
                metrics[
                    "accuracy"
                ]
        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=============================================="
    )

    print(
        "E2 - FULL CLEAN -> SIMULATED REPLAY SEEN"
    )

    print(
        "=============================================="
    )

    # ========================================================
    # INPUT CHECK
    # ========================================================

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            CHECKPOINT_PATH
        )

    if not TEST_MANIFEST.exists():

        raise FileNotFoundError(
            TEST_MANIFEST
        )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Best epoch:",
        checkpoint[
            "epoch"
        ]
    )

    clean_validation_eer = float(
        checkpoint[
            "validation_eer"
        ]
    )

    fixed_threshold = float(
        checkpoint[
            "validation_eer_threshold"
        ]
    )

    print(
        "Clean validation EER:",
        f"{clean_validation_eer * 100:.2f}%"
    )

    print(
        "Frozen validation threshold:",
        f"{fixed_threshold:.8f}"
    )

    # ========================================================
    # FEATURE SET SAFETY
    # ========================================================

    checkpoint_feature_set = (
        checkpoint.get(
            "feature_set",
            "full"
        )
    )

    if (
        checkpoint_feature_set
        !=
        "full"
    ):

        raise RuntimeError(
            "E2 membutuhkan checkpoint "
            "full feature."
        )

    # ========================================================
    # STATS
    #
    # Utamakan stats di dalam checkpoint supaya evaluator
    # tidak bergantung pada stats eksperimen lain.
    # ========================================================

    stats = checkpoint.get(
        "stats"
    )

    if stats is None:

        stats_path = Path(
            checkpoint[
                "stats_path"
            ]
        )

        if not stats_path.exists():

            raise FileNotFoundError(
                stats_path
            )

        stats = torch.load(
            stats_path,
            map_location="cpu",
            weights_only=False
        )

    print()
    print(
        "E2 training statistics:"
    )

    print(
        "Log-Mel:",
        stats[
            "logmel_mean"
        ],
        stats[
            "logmel_std"
        ]
    )

    print(
        "Phase:",
        stats[
            "phase_mean"
        ],
        stats[
            "phase_std"
        ]
    )

    print(
        "MGD:",
        stats[
            "mgd_mean"
        ],
        stats[
            "mgd_std"
        ]
    )

    # ========================================================
    # LOAD TEST MANIFEST
    # ========================================================

    df = pd.read_csv(
        TEST_MANIFEST
    )

    required = [
        "file_id",
        "source_file_id",
        "file_path",
        "label",
        "split",
        "condition",
        "rir_identity",
        "rir_group",
        "snr_db",
        "qc_status"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Manifest kehilangan kolom: {missing}"
        )

    # ========================================================
    # QC FILTER
    # ========================================================

    df[
        "label"
    ] = pd.to_numeric(
        df[
            "label"
        ],
        errors="raise"
    ).astype(int)

    test_df = df[
        (
            df[
                "split"
            ]
            .astype(str)
            .str.lower()
            ==
            "test"
        )
        &
        (
            df[
                "rir_group"
            ]
            .astype(str)
            .str.lower()
            ==
            "seen"
        )
        &
        (
            df[
                "qc_status"
            ]
            .astype(str)
            .str.upper()
            ==
            "PASS"
        )
    ].copy()

    print()
    print(
        "Test rows:",
        len(
            test_df
        )
    )

    print(
        "Unique source:",
        test_df[
            "source_file_id"
        ].nunique()
    )

    print(
        "Label:"
    )

    print(
        test_df[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "RIR:"
    )

    print(
        test_df[
            "rir_identity"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "SNR:"
    )

    print(
        test_df[
            "snr_db"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # STRICT AUDIT
    # ========================================================

    if len(
        test_df
    ) != 200:

        raise RuntimeError(
            f"E2 test harus 200, "
            f"ditemukan {len(test_df)}."
        )

    if (
        test_df[
            "source_file_id"
        ].nunique()
        !=
        200
    ):

        raise RuntimeError(
            "E2 test harus memiliki "
            "200 source unik."
        )

    if int(
        (
            test_df[
                "label"
            ]
            ==
            0
        ).sum()
    ) != 100:

        raise RuntimeError(
            "Label 0 harus 100."
        )

    if int(
        (
            test_df[
                "label"
            ]
            ==
            1
        ).sum()
    ) != 100:

        raise RuntimeError(
            "Label 1 harus 100."
        )

    # ========================================================
    # CHECK FILE EXISTENCE
    # ========================================================

    missing_files = [
        path
        for path in test_df[
            "file_path"
        ]
        if not Path(
            str(path)
        ).exists()
    ]

    print(
        "Missing audio:",
        len(
            missing_files
        )
    )

    if missing_files:

        raise RuntimeError(
            "Ada audio test yang hilang."
        )

    # ========================================================
    # FEATURE EXTRACTOR
    # ========================================================

    feature_extractor = SpectralPhase(
        sr=TARGET_SR,
        n_fft=512,
        hop=160,
        n_mels=80
    )

    feature_extractor.eval()

    # ========================================================
    # MODEL
    # ========================================================

    model = ReplayResNet(
        n_transform=2,
        n_domain=N_DOMAIN,
        embedding_dim=EMBEDDING_DIM
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    # ========================================================
    # INFERENCE
    # ========================================================

    score_rows = []

    total = len(
        test_df
    )

    print()
    print(
        "=============================================="
    )

    print(
        "INFERENCE"
    )

    print(
        "=============================================="
    )

    test_df = test_df.reset_index(
        drop=True
    )

    with torch.no_grad():

        for index, row in (
            test_df.iterrows()
        ):

            wav = load_audio_4s(
                row[
                    "file_path"
                ]
            )

            features = extract_features(
                wav,
                feature_extractor,
                stats
            )

            x = (
                features
                .unsqueeze(0)
                .to(device)
            )

            (
                class_logits,
                _,
                _,
                _
            ) = model(
                x,
                grl_strength=0.0
            )

            probability = F.softmax(
                class_logits,
                dim=1
            )

            fake_score = float(
                probability[
                    0,
                    1
                ].item()
            )

            score_rows.append({
                "file_id":
                    str(
                        row[
                            "file_id"
                        ]
                    ),

                "source_file_id":
                    str(
                        row[
                            "source_file_id"
                        ]
                    ),

                "label":
                    int(
                        row[
                            "label"
                        ]
                    ),

                "fake_score":
                    fake_score,

                "fixed_threshold":
                    fixed_threshold,

                "condition":
                    row[
                        "condition"
                    ],

                "rir_identity":
                    row[
                        "rir_identity"
                    ],

                "rir_group":
                    row[
                        "rir_group"
                    ],

                "snr_db":
                    int(
                        row[
                            "snr_db"
                        ]
                    )
            })

            if (
                (index + 1) % 20 == 0
                or
                (index + 1) == total
            ):

                print(
                    f"[{index + 1}/{total}]"
                )

    scores_df = pd.DataFrame(
        score_rows
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    metrics = compute_metrics(
        scores_df[
            "label"
        ].values,
        scores_df[
            "fake_score"
        ].values,
        fixed_threshold
    )

    simulated_eer = float(
        metrics[
            "eer"
        ]
    )

    clean_to_replay_gap = (
        simulated_eer
        -
        clean_validation_eer
    )

    # ========================================================
    # CONDITION ANALYSIS
    # ========================================================

    condition_rows = []

    condition_rows.extend(
        condition_metrics(
            scores_df,
            "rir_identity",
            fixed_threshold
        )
    )

    condition_rows.extend(
        condition_metrics(
            scores_df,
            "snr_db",
            fixed_threshold
        )
    )

    conditions_df = pd.DataFrame(
        condition_rows
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "experiment":
                    "E2",

                "training":
                    "full_feature_clean_only",

                "test":
                    "simulated_replay_seen_source_disjoint",

                "n":
                    len(
                        scores_df
                    ),

                "clean_validation_eer":
                    clean_validation_eer,

                "simulated_replay_eer":
                    simulated_eer,

                "clean_to_replay_gap":
                    clean_to_replay_gap,

                "clean_to_replay_gap_pp":
                    clean_to_replay_gap
                    *
                    100.0,

                "auc":
                    metrics[
                        "auc"
                    ],

                "f1_fixed_clean_threshold":
                    metrics[
                        "f1"
                    ],

                "balanced_accuracy_fixed_clean_threshold":
                    metrics[
                        "balanced_accuracy"
                    ],

                "accuracy_fixed_clean_threshold":
                    metrics[
                        "accuracy"
                    ],

                "clean_validation_threshold":
                    fixed_threshold,

                "test_eer_threshold":
                    metrics[
                        "test_eer_threshold"
                    ]
            }
        ]
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_SCORES.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    scores_df.to_csv(
        OUTPUT_SCORES,
        index=False
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    conditions_df.to_csv(
        OUTPUT_CONDITIONS,
        index=False
    )

    config = {
        "experiment":
            "E2",

        "checkpoint":
            str(
                CHECKPOINT_PATH
            ),

        "checkpoint_epoch":
            int(
                checkpoint[
                    "epoch"
                ]
            ),

        "test_manifest":
            str(
                TEST_MANIFEST
            ),

        "test_rows":
            len(
                scores_df
            ),

        "feature_set":
            "full",

        "training_condition":
            "clean_only",

        "test_condition":
            "simulated_replay_seen",

        "source_disjoint":
            True,

        "clean_validation_eer":
            clean_validation_eer,

        "clean_validation_threshold":
            fixed_threshold,

        "simulated_replay_eer":
            simulated_eer,

        "clean_to_replay_gap":
            clean_to_replay_gap
    }

    with open(
        OUTPUT_CONFIG,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    # ========================================================
    # PRINT FINAL
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "E2 EVALUATION SELESAI"
    )

    print(
        "=============================================="
    )

    print()
    print(
        "Clean validation:"
    )

    print(
        "EER              :",
        f"{clean_validation_eer * 100:.2f}%"
    )

    print(
        "Frozen threshold :",
        f"{fixed_threshold:.8f}"
    )

    print()
    print(
        "Simulated replay seen:"
    )

    print(
        "EER              :",
        f"{metrics['eer'] * 100:.2f}%"
    )

    print(
        "AUC              :",
        f"{metrics['auc']:.4f}"
    )

    print(
        "F1               :",
        f"{metrics['f1']:.4f}"
    )

    print(
        "Balanced accuracy:",
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        "Accuracy         :",
        f"{metrics['accuracy']:.4f}"
    )

    print()
    print(
        "CLEAN-TO-REPLAY GAP"
    )

    print(
        f"{clean_validation_eer * 100:.2f}%"
        " -> "
        f"{simulated_eer * 100:.2f}%"
    )

    print(
        "Gap:",
        f"{clean_to_replay_gap * 100:+.2f} "
        "percentage points"
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_SCORES
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_CONDITIONS
    )

    print(
        OUTPUT_CONFIG
    )

if __name__ == "__main__":

    main()