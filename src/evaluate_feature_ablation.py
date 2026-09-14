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
# CONFIG
# ============================================================

TARGET_SR = 16000
SECONDS = 4

N_DOMAIN = 5
EMBEDDING_DIM = 256

SIM_UNSEEN_MANIFEST = Path(
    "manifests/manifest_replay_sim_week9.csv"
)

REAL_REPLAY_MANIFEST = Path(
    "manifests/real_replay_manifest_480.csv"
)

OUTPUT_SUMMARY = Path(
    "results/feature_ablation_summary.csv"
)

OUTPUT_SCORES = Path(
    "results/feature_ablation_scores.csv"
)

OUTPUT_CONFIG = Path(
    "results/feature_ablation_config.json"
)

REPORT_TABLE_PATH = Path(
    "results/feature_ablation_report_table.csv"
)


# ============================================================
# EXPERIMENTS
# ============================================================

EXPERIMENTS = {

    "logmel": {

        "display_name":
            "Log-Mel only",

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_ablation_logmel_no_grl.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_ablation_logmel_no_grl_validation.csv"
            ),

        "feature_set":
            "logmel"
    },

    "logmel_phase": {

        "display_name":
            "Log-Mel + Phase",

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_ablation_logmel_phase_no_grl.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_ablation_logmel_phase_no_grl_validation.csv"
            ),

        "feature_set":
            "logmel_phase"
    },

    "full": {

        "display_name":
            "Full three-channel",

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_no_grl.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_no_grl_validation.csv"
            ),

        "feature_set":
            "full"
    }
}


# ============================================================
# AUDIO LOADER
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

    # --------------------------------------------------------
    # Multi-channel -> mono
    # --------------------------------------------------------

    audio = audio.mean(
        axis=1
    )

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Pad
    # --------------------------------------------------------

    if n < target_length:

        audio = np.pad(
            audio,
            (
                0,
                target_length - n
            ),
            mode="constant"
        )

    # --------------------------------------------------------
    # Center crop
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Peak normalization
    # Sama dengan training
    # --------------------------------------------------------

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

    wav = (
        torch
        .from_numpy(
            audio
        )
        .float()
        .unsqueeze(0)
    )

    return wav


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    wav,
    extractor,
    stats,
    feature_set
):

    # --------------------------------------------------------
    # [1,T] -> [1,1,T]
    # --------------------------------------------------------

    wav = wav.unsqueeze(0)

    with torch.no_grad():

        logmel, phase = (
            extractor(
                wav
            )
        )

        mgd = modified_group_delay(
            wav,
            extractor.fb
        )

    # --------------------------------------------------------
    # [1,80,T] -> [80,T]
    # --------------------------------------------------------

    logmel = logmel.squeeze(0)
    phase = phase.squeeze(0)
    mgd = mgd.squeeze(0)

    # --------------------------------------------------------
    # Finite check
    # --------------------------------------------------------

    for name, feature in [

        (
            "logmel",
            logmel
        ),

        (
            "phase",
            phase
        ),

        (
            "mgd",
            mgd
        )
    ]:

        if not torch.isfinite(
            feature
        ).all():

            raise RuntimeError(
                f"{name} memiliki NaN/Inf."
            )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Zero channels
    # --------------------------------------------------------

    zero_phase = torch.zeros_like(
        phase
    )

    zero_mgd = torch.zeros_like(
        mgd
    )

    # --------------------------------------------------------
    # Ablation configuration
    # --------------------------------------------------------

    if feature_set == "logmel":

        features = torch.stack(
            [
                logmel,
                zero_phase,
                zero_mgd
            ],
            dim=0
        )

    elif feature_set == "logmel_phase":

        features = torch.stack(
            [
                logmel,
                phase,
                zero_mgd
            ],
            dim=0
        )

    elif feature_set == "full":

        features = torch.stack(
            [
                logmel,
                phase,
                mgd
            ],
            dim=0
        )

    else:

        raise ValueError(
            f"Feature set tidak dikenal: "
            f"{feature_set}"
        )

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
# METRICS
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
            "Metrics membutuhkan label 0 dan 1."
        )

    eer, eer_threshold = (
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

    prediction = (
        scores
        >=
        fixed_threshold
    ).astype(int)

    f1 = float(
        f1_score(
            labels,
            prediction,
            zero_division=0
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            labels,
            prediction
        )
    )

    accuracy = float(
        accuracy_score(
            labels,
            prediction
        )
    )

    return {

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
            accuracy
    }


# ============================================================
# LOAD VALIDATION SCORES
# ============================================================

def load_validation_scores(
    validation_scores_path
):

    if not validation_scores_path.exists():

        raise FileNotFoundError(
            validation_scores_path
        )

    validation_scores = pd.read_csv(
        validation_scores_path
    )

    required = [
        "label",
        "fake_score"
    ]

    missing = [

        column

        for column in required

        if column
        not in validation_scores.columns
    ]

    if missing:

        raise RuntimeError(
            "Validation score CSV kehilangan "
            f"kolom: {missing}"
        )

    validation_scores[
        "label"
    ] = pd.to_numeric(
        validation_scores[
            "label"
        ],
        errors="raise"
    ).astype(int)

    validation_scores[
        "fake_score"
    ] = pd.to_numeric(
        validation_scores[
            "fake_score"
        ],
        errors="raise"
    ).astype(float)

    return validation_scores


# ============================================================
# LOAD EXPERIMENT
# ============================================================

def load_experiment(
    experiment,
    device
):

    checkpoint_path = (
        experiment[
            "checkpoint"
        ]
    )

    validation_scores_path = (
        experiment[
            "validation_scores"
        ]
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

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
    # STATS
    # ========================================================

    stats = checkpoint.get(
        "stats"
    )

    if stats is None:

        stats_path_value = checkpoint.get(
            "stats_path"
        )

        if stats_path_value is not None:

            stats_path = Path(
                stats_path_value
            )

        else:

            # ------------------------------------------------
            # Fallback checkpoint lama
            # ------------------------------------------------

            stats_path = Path(
                "data/processed/"
                "three_channel_stats.pt"
            )

        if not stats_path.exists():

            raise FileNotFoundError(
                f"Stats tidak ditemukan: "
                f"{stats_path}"
            )

        stats = torch.load(
            stats_path,
            map_location="cpu",
            weights_only=False
        )

    # ========================================================
    # VALIDATION EER
    # ========================================================

    if (
        "validation_eer"
        in checkpoint
    ):

        validation_eer = float(
            checkpoint[
                "validation_eer"
            ]
        )

    elif (
        "best_eer"
        in checkpoint
    ):

        validation_eer = float(
            checkpoint[
                "best_eer"
            ]
        )

    else:

        validation_scores = (
            load_validation_scores(
                validation_scores_path
            )
        )

        validation_eer, _ = (
            calculate_eer(
                validation_scores[
                    "label"
                ].values,
                validation_scores[
                    "fake_score"
                ].values
            )
        )

    # ========================================================
    # VALIDATION THRESHOLD
    # ========================================================

    if (
        "validation_eer_threshold"
        in checkpoint
    ):

        fixed_threshold = float(
            checkpoint[
                "validation_eer_threshold"
            ]
        )

    else:

        print(
            "Checkpoint lama tidak memiliki "
            "validation_eer_threshold."
        )

        print(
            "Mencari threshold dari:",
            validation_scores_path
        )

        validation_scores = (
            load_validation_scores(
                validation_scores_path
            )
        )

        # ----------------------------------------------------
        # Format baru
        # ----------------------------------------------------

        if (
            "validation_eer_threshold"
            in validation_scores.columns
        ):

            threshold_values = (
                validation_scores[
                    "validation_eer_threshold"
                ]
                .dropna()
                .unique()
            )

            if len(
                threshold_values
            ) == 0:

                raise RuntimeError(
                    "validation_eer_threshold kosong."
                )

            fixed_threshold = float(
                threshold_values[
                    0
                ]
            )

        # ----------------------------------------------------
        # Alternate format
        # ----------------------------------------------------

        elif (
            "eer_threshold"
            in validation_scores.columns
        ):

            threshold_values = (
                validation_scores[
                    "eer_threshold"
                ]
                .dropna()
                .unique()
            )

            if len(
                threshold_values
            ) == 0:

                raise RuntimeError(
                    "eer_threshold kosong."
                )

            fixed_threshold = float(
                threshold_values[
                    0
                ]
            )

        # ----------------------------------------------------
        # Old score CSV
        # ----------------------------------------------------

        else:

            _, fixed_threshold = (
                calculate_eer(
                    validation_scores[
                        "label"
                    ].values,
                    validation_scores[
                        "fake_score"
                    ].values
                )
            )

            print(
                "Threshold dihitung ulang "
                "dari validation continuous scores."
            )

    return (
        model,
        checkpoint,
        stats,
        validation_eer,
        fixed_threshold
    )


# ============================================================
# SIMULATED UNSEEN
# ============================================================

def load_simulated_unseen():

    if not SIM_UNSEEN_MANIFEST.exists():

        raise FileNotFoundError(
            SIM_UNSEEN_MANIFEST
        )

    df = pd.read_csv(
        SIM_UNSEEN_MANIFEST
    )

    required = [
        "file_path",
        "source_file_id",
        "label",
        "split",
        "rir_group",
        "rir_identity",
        "snr_db"
    ]

    missing = [

        column

        for column in required

        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Sim manifest kehilangan "
            f"kolom: {missing}"
        )

    df[
        "label"
    ] = pd.to_numeric(
        df[
            "label"
        ],
        errors="raise"
    ).astype(int)

    # --------------------------------------------------------
    # TEST + UNSEEN RIR
    # --------------------------------------------------------

    test = df[
        (
            df[
                "split"
            ]
            .astype(str)
            .str.strip()
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
            .str.strip()
            .str.lower()
            ==
            "unseen"
        )
    ].copy()

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    if (
        "qc_status"
        in test.columns
    ):

        test = test[
            test[
                "qc_status"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            "PASS"
        ].copy()

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    if len(
        test
    ) != 200:

        raise RuntimeError(
            "Sim unseen seharusnya 200, "
            f"ditemukan {len(test)}."
        )

    if int(
        (
            test[
                "label"
            ]
            ==
            0
        ).sum()
    ) != 100:

        raise RuntimeError(
            "Sim unseen label 0 harus 100."
        )

    if int(
        (
            test[
                "label"
            ]
            ==
            1
        ).sum()
    ) != 100:

        raise RuntimeError(
            "Sim unseen label 1 harus 100."
        )

    # --------------------------------------------------------
    # Audio existence
    # --------------------------------------------------------

    missing_audio = [

        path

        for path in test[
            "file_path"
        ]

        if not Path(
            str(path)
        ).exists()
    ]

    if missing_audio:

        raise RuntimeError(
            "Ada simulated unseen audio "
            "yang tidak ditemukan."
        )

    return (
        test
        .reset_index(
            drop=True
        )
    )


# ============================================================
# REAL REPLAY
# ============================================================

def load_real_replay():

    if not REAL_REPLAY_MANIFEST.exists():

        raise FileNotFoundError(
            REAL_REPLAY_MANIFEST
        )

    df = pd.read_csv(
        REAL_REPLAY_MANIFEST
    )

    # ========================================================
    # AUDIO PATH
    # ========================================================

    path_column = None

    for candidate in [
        "recording_path",
        "file_path",
        "output_path"
    ]:

        if candidate in df.columns:

            path_column = (
                candidate
            )

            break

    if path_column is None:

        raise RuntimeError(
            "Kolom path real replay "
            "tidak ditemukan."
        )

    required = [
        "source_file_id",
        "label",
        "mic"
    ]

    missing = [

        column

        for column in required

        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Real manifest kehilangan "
            f"kolom: {missing}"
        )

    df[
        "label"
    ] = pd.to_numeric(
        df[
            "label"
        ],
        errors="raise"
    ).astype(int)

    # ========================================================
    # QC
    # ========================================================

    if (
        "qc_status"
        in df.columns
    ):

        df = df[
            df[
                "qc_status"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            "PASS"
        ].copy()

    if (
        "recording_status"
        in df.columns
    ):

        df = df[
            df[
                "recording_status"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            "DONE"
        ].copy()

    df[
        "_audio_path"
    ] = df[
        path_column
    ]

    # ========================================================
    # AUDIT
    # ========================================================

    if len(
        df
    ) != 480:

        raise RuntimeError(
            "Real replay seharusnya 480, "
            f"ditemukan {len(df)}."
        )

    if int(
        (
            df[
                "label"
            ]
            ==
            0
        ).sum()
    ) != 240:

        raise RuntimeError(
            "Real label 0 harus 240."
        )

    if int(
        (
            df[
                "label"
            ]
            ==
            1
        ).sum()
    ) != 240:

        raise RuntimeError(
            "Real label 1 harus 240."
        )

    # --------------------------------------------------------
    # Audio existence
    # --------------------------------------------------------

    missing_audio = [

        path

        for path in df[
            "_audio_path"
        ]

        if not Path(
            str(path)
        ).exists()
    ]

    if missing_audio:

        raise RuntimeError(
            "Ada real replay audio "
            "yang tidak ditemukan."
        )

    return (
        df
        .reset_index(
            drop=True
        )
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def evaluate_dataframe(
    model,
    dataframe,
    audio_path_column,
    feature_extractor,
    stats,
    feature_set,
    device,
    fixed_threshold,
    model_name,
    test_condition
):

    rows = []

    total = len(
        dataframe
    )

    for index, row in (
        dataframe.iterrows()
    ):

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        wav = load_audio_4s(
            row[
                audio_path_column
            ]
        )

        # ----------------------------------------------------
        # Feature
        # ----------------------------------------------------

        features = extract_features(
            wav,
            feature_extractor,
            stats,
            feature_set
        )

        x = (
            features
            .unsqueeze(0)
            .to(device)
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

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

        output = {

            "model":
                model_name,

            "feature_set":
                feature_set,

            "test_condition":
                test_condition,

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
                fixed_threshold
        }

        # ----------------------------------------------------
        # Optional metadata
        # ----------------------------------------------------

        optional_columns = [

            "file_id",
            "recording_id",

            "rir_identity",
            "rir_group",
            "snr_db",

            "room",
            "mic",
            "playback",

            "distance",
            "angle",
            "repetition"
        ]

        for column in optional_columns:

            if column in row.index:

                output[
                    column
                ] = row[
                    column
                ]

        rows.append(
            output
        )

        if (
            (index + 1) % 20 == 0
            or
            (index + 1) == total
        ):

            print(
                f"[{index + 1}/{total}]"
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary_row(
    model_name,
    feature_set,
    condition,
    dataframe,
    fixed_threshold
):

    if (
        dataframe[
            "label"
        ].nunique()
        !=
        2
    ):

        raise RuntimeError(
            f"{condition} tidak memiliki "
            "dua kelas."
        )

    metrics = compute_metrics(
        dataframe[
            "label"
        ].values,
        dataframe[
            "fake_score"
        ].values,
        fixed_threshold
    )

    return {

        "model":
            model_name,

        "feature_set":
            feature_set,

        "condition":
            condition,

        "n":
            len(
                dataframe
            ),

        "label_0":
            int(
                (
                    dataframe[
                        "label"
                    ]
                    ==
                    0
                ).sum()
            ),

        "label_1":
            int(
                (
                    dataframe[
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
            ],

        "fixed_threshold":
            fixed_threshold,

        "test_eer_threshold":
            metrics[
                "eer_threshold"
            ]
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=============================================="
    )

    print(
        "TAHAP 8 - FEATURE ABLATION EVALUATION"
    )

    print(
        "=============================================="
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
    # DATASETS
    # ========================================================

    sim_unseen = (
        load_simulated_unseen()
    )

    real_replay = (
        load_real_replay()
    )

    print()
    print(
        "Simulated unseen rows:",
        len(
            sim_unseen
        )
    )

    print(
        "Real replay rows:",
        len(
            real_replay
        )
    )

    print(
        "Real mic distribution:"
    )

    print(
        real_replay[
            "mic"
        ]
        .value_counts()
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
    # ACCUMULATORS
    # ========================================================

    all_scores = []
    summary_rows = []

    # ========================================================
    # LOOP EXPERIMENTS
    # ========================================================

    for key, experiment in (
        EXPERIMENTS.items()
    ):

        model_name = (
            experiment[
                "display_name"
            ]
        )

        feature_set = (
            experiment[
                "feature_set"
            ]
        )

        print()
        print(
            "=============================================="
        )

        print(
            model_name
        )

        print(
            "=============================================="
        )

        (
            model,
            checkpoint,
            stats,
            validation_eer,
            fixed_threshold
        ) = load_experiment(
            experiment,
            device
        )

        print(
            "Checkpoint:",
            experiment[
                "checkpoint"
            ]
        )

        print(
            "Feature set:",
            feature_set
        )

        print(
            "Best epoch:",
            checkpoint.get(
                "epoch",
                "unknown"
            )
        )

        print(
            "Validation EER:",
            f"{validation_eer * 100:.2f}%"
        )

        print(
            "Frozen threshold:",
            f"{fixed_threshold:.8f}"
        )

        # ====================================================
        # VALIDATION SEEN
        # ====================================================

        validation_scores_path = (
            experiment[
                "validation_scores"
            ]
        )

        validation_scores_df = (
            load_validation_scores(
                validation_scores_path
            )
        )

        validation_metrics = (
            compute_metrics(
                validation_scores_df[
                    "label"
                ].values,
                validation_scores_df[
                    "fake_score"
                ].values,
                fixed_threshold
            )
        )

        validation_summary = {

            "model":
                model_name,

            "feature_set":
                feature_set,

            "condition":
                "validation_seen",

            "n":
                len(
                    validation_scores_df
                ),

            "label_0":
                int(
                    (
                        validation_scores_df[
                            "label"
                        ]
                        ==
                        0
                    ).sum()
                ),

            "label_1":
                int(
                    (
                        validation_scores_df[
                            "label"
                        ]
                        ==
                        1
                    ).sum()
                ),

            "eer":
                validation_metrics[
                    "eer"
                ],

            "auc":
                validation_metrics[
                    "auc"
                ],

            "f1":
                validation_metrics[
                    "f1"
                ],

            "balanced_accuracy":
                validation_metrics[
                    "balanced_accuracy"
                ],

            "accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "fixed_threshold":
                fixed_threshold,

            "test_eer_threshold":
                validation_metrics[
                    "eer_threshold"
                ]
        }

        summary_rows.append(
            validation_summary
        )

        # ====================================================
        # SIMULATED UNSEEN
        # ====================================================

        print()
        print(
            "Simulated unseen RIR:"
        )

        sim_scores = (
            evaluate_dataframe(

                model=
                    model,

                dataframe=
                    sim_unseen,

                audio_path_column=
                    "file_path",

                feature_extractor=
                    feature_extractor,

                stats=
                    stats,

                feature_set=
                    feature_set,

                device=
                    device,

                fixed_threshold=
                    fixed_threshold,

                model_name=
                    model_name,

                test_condition=
                    "simulated_unseen_rir"
            )
        )

        all_scores.append(
            sim_scores
        )

        summary_rows.append(
            build_summary_row(
                model_name,
                feature_set,
                "simulated_unseen_rir",
                sim_scores,
                fixed_threshold
            )
        )

        # ====================================================
        # REAL REPLAY
        # ====================================================

        print()
        print(
            "Controlled real replay:"
        )

        real_scores = (
            evaluate_dataframe(

                model=
                    model,

                dataframe=
                    real_replay,

                audio_path_column=
                    "_audio_path",

                feature_extractor=
                    feature_extractor,

                stats=
                    stats,

                feature_set=
                    feature_set,

                device=
                    device,

                fixed_threshold=
                    fixed_threshold,

                model_name=
                    model_name,

                test_condition=
                    "real_replay"
            )
        )

        all_scores.append(
            real_scores
        )

        summary_rows.append(
            build_summary_row(
                model_name,
                feature_set,
                "real_replay",
                real_scores,
                fixed_threshold
            )
        )

        # ====================================================
        # MIC UNSEEN
        # ====================================================

        mic_unseen = real_scores[
            real_scores[
                "mic"
            ]
            .astype(str)
            .str.strip()
            ==
            "mic_unseen"
        ].copy()

        if len(
            mic_unseen
        ) != 240:

            raise RuntimeError(
                f"{model_name}: "
                "mic_unseen seharusnya 240, "
                f"ditemukan {len(mic_unseen)}."
            )

        summary_rows.append(
            build_summary_row(
                model_name,
                feature_set,
                "real_replay_mic_unseen",
                mic_unseen,
                fixed_threshold
            )
        )

    # ========================================================
    # CONCAT
    # ========================================================

    scores_df = pd.concat(
        all_scores,
        ignore_index=True
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # REPORT-READY EER TABLE
    # ========================================================

    report_table = (
        summary_df
        .pivot(
            index=[
                "model",
                "feature_set"
            ],
            columns="condition",
            values="eer"
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Convert EER -> percent
    # --------------------------------------------------------

    eer_columns = [

        "validation_seen",

        "simulated_unseen_rir",

        "real_replay",

        "real_replay_mic_unseen"
    ]

    for column in eer_columns:

        if column in report_table.columns:

            report_table[
                column
            ] = (
                report_table[
                    column
                ]
                *
                100.0
            )

    # --------------------------------------------------------
    # Rename report columns
    # --------------------------------------------------------

    report_table = (
        report_table.rename(

            columns={

                "validation_seen":
                    "validation_seen_eer_percent",

                "simulated_unseen_rir":
                    "sim_unseen_eer_percent",

                "real_replay":
                    "real_replay_eer_percent",

                "real_replay_mic_unseen":
                    "unseen_mic_eer_percent"
            }
        )
    )

    # --------------------------------------------------------
    # Desired report ordering
    # --------------------------------------------------------

    feature_order = {

        "logmel":
            0,

        "logmel_phase":
            1,

        "full":
            2
    }

    report_table[
        "_order"
    ] = (
        report_table[
            "feature_set"
        ]
        .map(
            feature_order
        )
    )

    report_table = (
        report_table
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Round report values
    # --------------------------------------------------------

    numeric_report_columns = [

        "validation_seen_eer_percent",

        "sim_unseen_eer_percent",

        "real_replay_eer_percent",

        "unseen_mic_eer_percent"
    ]

    for column in numeric_report_columns:

        if column in report_table.columns:

            report_table[
                column
            ] = (
                report_table[
                    column
                ]
                .round(
                    2
                )
            )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    scores_df.to_csv(
        OUTPUT_SCORES,
        index=False
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    report_table.to_csv(
        REPORT_TABLE_PATH,
        index=False
    )

    # ========================================================
    # CONFIG
    # ========================================================

    config = {

        "experiments": {

            key: {

                "display_name":
                    value[
                        "display_name"
                    ],

                "checkpoint":
                    str(
                        value[
                            "checkpoint"
                        ]
                    ),

                "validation_scores":
                    str(
                        value[
                            "validation_scores"
                        ]
                    ),

                "feature_set":
                    value[
                        "feature_set"
                    ]
            }

            for key, value
            in EXPERIMENTS.items()
        },

        "simulated_unseen_manifest":
            str(
                SIM_UNSEEN_MANIFEST
            ),

        "real_replay_manifest":
            str(
                REAL_REPLAY_MANIFEST
            ),

        "conditions": [

            "validation_seen",

            "simulated_unseen_rir",

            "real_replay",

            "real_replay_mic_unseen"
        ],

        "metric_policy": {

            "eer":
                "continuous scores",

            "auc":
                "continuous scores",

            "f1":
                "frozen validation EER threshold",

            "balanced_accuracy":
                "frozen validation EER threshold",

            "accuracy":
                "frozen validation EER threshold"
        },

        "report_table":
            str(
                REPORT_TABLE_PATH
            )
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
    # PRINT SUMMARY
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "FEATURE ABLATION SUMMARY"
    )

    print(
        "=============================================="
    )

    conditions = [

        "validation_seen",

        "simulated_unseen_rir",

        "real_replay",

        "real_replay_mic_unseen"
    ]

    for condition in conditions:

        print()
        print(
            "Condition:",
            condition
        )

        subset = summary_df[
            summary_df[
                "condition"
            ]
            ==
            condition
        ]

        for _, row in (
            subset.iterrows()
        ):

            print(
                f"{row['model']:22s} | "
                f"n={int(row['n']):3d} | "
                f"EER={row['eer'] * 100:6.2f}% | "
                f"AUC={row['auc']:.4f} | "
                f"F1={row['f1']:.4f} | "
                f"BAcc="
                f"{row['balanced_accuracy']:.4f}"
            )

    # ========================================================
    # FEATURE CONTRIBUTION
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "FEATURE CONTRIBUTION"
    )

    print(
        "=============================================="
    )

    for condition in conditions:

        subset = (
            summary_df[
                summary_df[
                    "condition"
                ]
                ==
                condition
            ]
            .set_index(
                "feature_set"
            )
        )

        required_feature_sets = [
            "logmel",
            "logmel_phase",
            "full"
        ]

        if not all(
            feature
            in subset.index

            for feature
            in required_feature_sets
        ):

            continue

        eer_logmel = float(
            subset.loc[
                "logmel",
                "eer"
            ]
        )

        eer_phase = float(
            subset.loc[
                "logmel_phase",
                "eer"
            ]
        )

        eer_full = float(
            subset.loc[
                "full",
                "eer"
            ]
        )

        # ----------------------------------------------------
        # Positive = improvement
        # Negative = degradation
        # ----------------------------------------------------

        phase_delta = (
            eer_logmel
            -
            eer_phase
        ) * 100.0

        mgd_delta = (
            eer_phase
            -
            eer_full
        ) * 100.0

        total_delta = (
            eer_logmel
            -
            eer_full
        ) * 100.0

        print()
        print(
            condition
        )

        print(
            "Log-Mel EER       :",
            f"{eer_logmel * 100:.2f}%"
        )

        print(
            "Log-Mel+Phase EER :",
            f"{eer_phase * 100:.2f}%"
        )

        print(
            "Full EER          :",
            f"{eer_full * 100:.2f}%"
        )

        print(
            "Phase contribution:",
            f"{phase_delta:+.2f} pp"
        )

        print(
            "MGD contribution  :",
            f"{mgd_delta:+.2f} pp"
        )

        print(
            "Full vs Log-Mel   :",
            f"{total_delta:+.2f} pp"
        )

    # ========================================================
    # REPORT TABLE
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "REPORT-READY EER TABLE"
    )

    print(
        "=============================================="
    )

    print()

    print(
        report_table.to_string(
            index=False
        )
    )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "FEATURE ABLATION EVALUATION SELESAI"
    )

    print(
        "=============================================="
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_SCORES
    )

    print(
        OUTPUT_CONFIG
    )

    print(
        REPORT_TABLE_PATH
    )

if __name__ == "__main__":
    main()