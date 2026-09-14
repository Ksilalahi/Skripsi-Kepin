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
# GLOBAL CONFIG
# ============================================================

TARGET_SR = 16000
SECONDS = 4

N_DOMAIN = 5
EMBEDDING_DIM = 256

SEEDS = [
    2026,
    2027,
    2028
]

SIM_UNSEEN_MANIFEST = Path(
    "manifests/manifest_replay_sim_week9.csv"
)

REAL_REPLAY_MANIFEST = Path(
    "manifests/real_replay_manifest_480.csv"
)

RESULTS_DIR = Path(
    "results"
)

OUTPUT_SCORES = RESULTS_DIR / "scores.csv"

OUTPUT_RUN_SUMMARY = (
    RESULTS_DIR
    /
    "run_summary.csv"
)

OUTPUT_CONDITION_SUMMARY = (
    RESULTS_DIR
    /
    "multiseed_condition_summary.csv"
)

OUTPUT_REPORT_TABLE = (
    RESULTS_DIR
    /
    "multiseed_report_table.csv"
)

OUTPUT_CONFIG = (
    RESULTS_DIR
    /
    "multiseed_config.json"
)


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================
#
# Seed 2026 memakai nama eksperimen lama.
# Seed 2027 dan 2028 memakai nama baru agar checkpoint
# tidak saling menimpa.
#
# ============================================================

EXPERIMENTS = [

    # --------------------------------------------------------
    # FULL NO-GRL
    # --------------------------------------------------------

    {
        "model_name":
            "Full No-GRL",

        "feature_set":
            "full",

        "use_grl":
            False,

        "seed":
            2026,

        "checkpoint":
            Path(
                "checkpoints/replay_no_grl.pt"
            ),

        "validation_scores":
            Path(
                "results/scores_no_grl_validation.csv"
            )
    },

    {
        "model_name":
            "Full No-GRL",

        "feature_set":
            "full",

        "use_grl":
            False,

        "seed":
            2027,

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_full_no_grl_seed2027.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_full_no_grl_seed2027_validation.csv"
            )
    },

    {
        "model_name":
            "Full No-GRL",

        "feature_set":
            "full",

        "use_grl":
            False,

        "seed":
            2028,

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_full_no_grl_seed2028.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_full_no_grl_seed2028_validation.csv"
            )
    },

    # --------------------------------------------------------
    # FULL + GRL
    # --------------------------------------------------------

    {
        "model_name":
            "Full + GRL",

        "feature_set":
            "full",

        "use_grl":
            True,

        "seed":
            2026,

        "checkpoint":
            Path(
                "checkpoints/replay_with_grl.pt"
            ),

        "validation_scores":
            Path(
                "results/scores_with_grl_validation.csv"
            )
    },

    {
        "model_name":
            "Full + GRL",

        "feature_set":
            "full",

        "use_grl":
            True,

        "seed":
            2027,

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_full_with_grl_seed2027.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_full_with_grl_seed2027_validation.csv"
            )
    },

    {
        "model_name":
            "Full + GRL",

        "feature_set":
            "full",

        "use_grl":
            True,

        "seed":
            2028,

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_full_with_grl_seed2028.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_full_with_grl_seed2028_validation.csv"
            )
    },

    # --------------------------------------------------------
    # LOG-MEL ONLY
    # --------------------------------------------------------

    {
        "model_name":
            "Log-Mel only",

        "feature_set":
            "logmel",

        "use_grl":
            False,

        "seed":
            2026,

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_ablation_logmel_no_grl.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_ablation_logmel_no_grl_validation.csv"
            )
    },

    {
        "model_name":
            "Log-Mel only",

        "feature_set":
            "logmel",

        "use_grl":
            False,

        "seed":
            2027,

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_ablation_logmel_seed2027.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_ablation_logmel_seed2027_validation.csv"
            )
    },

    {
        "model_name":
            "Log-Mel only",

        "feature_set":
            "logmel",

        "use_grl":
            False,

        "seed":
            2028,

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_ablation_logmel_seed2028.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_ablation_logmel_seed2028_validation.csv"
            )
    },

    # --------------------------------------------------------
    # LOG-MEL + PHASE
    # --------------------------------------------------------

    {
        "model_name":
            "Log-Mel + Phase",

        "feature_set":
            "logmel_phase",

        "use_grl":
            False,

        "seed":
            2026,

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_ablation_logmel_phase_no_grl.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_ablation_logmel_phase_no_grl_validation.csv"
            )
    },

    {
        "model_name":
            "Log-Mel + Phase",

        "feature_set":
            "logmel_phase",

        "use_grl":
            False,

        "seed":
            2027,

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_ablation_logmel_phase_seed2027.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_ablation_logmel_phase_seed2027_validation.csv"
            )
    },

    {
        "model_name":
            "Log-Mel + Phase",

        "feature_set":
            "logmel_phase",

        "use_grl":
            False,

        "seed":
            2028,

        "checkpoint":
            Path(
                "checkpoints/"
                "replay_ablation_logmel_phase_seed2028.pt"
            ),

        "validation_scores":
            Path(
                "results/"
                "scores_ablation_logmel_phase_seed2028_validation.csv"
            )
    }
]


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

    # Multi-channel -> mono
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

    # Peak normalization
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

    # [1,T] -> [1,1,T]
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

    logmel = logmel.squeeze(0)
    phase = phase.squeeze(0)
    mgd = mgd.squeeze(0)

    # --------------------------------------------------------
    # Finite check
    # --------------------------------------------------------

    for name, feature in [

        ("logmel", logmel),
        ("phase", phase),
        ("mgd", mgd)
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

    zero_phase = torch.zeros_like(
        phase
    )

    zero_mgd = torch.zeros_like(
        mgd
    )

    # --------------------------------------------------------
    # Feature ablation
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

    bacc = float(
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
            bacc,

        "accuracy":
            accuracy
    }


# ============================================================
# VALIDATION SCORES
# ============================================================

def load_validation_scores(
    path
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path
    )

    required = [
        "label",
        "fake_score"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{path} kehilangan "
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

    df[
        "fake_score"
    ] = pd.to_numeric(
        df[
            "fake_score"
        ],
        errors="raise"
    ).astype(float)

    return df


# ============================================================
# CHECKPOINT LOADER
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

    validation_path = (
        experiment[
            "validation_scores"
        ]
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Support old/new checkpoint key
    # --------------------------------------------------------

    if (
        "model_state_dict"
        in checkpoint
    ):

        model_state = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif (
        "model_state"
        in checkpoint
    ):

        model_state = (
            checkpoint[
                "model_state"
            ]
        )

    else:

        raise RuntimeError(
            f"{checkpoint_path} tidak memiliki "
            "model_state_dict/model_state."
        )

    model = ReplayResNet(
        n_transform=2,
        n_domain=N_DOMAIN,
        embedding_dim=EMBEDDING_DIM
    )

    model.load_state_dict(
        model_state
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

        stats_path_value = (
            checkpoint.get(
                "stats_path"
            )
        )

        if stats_path_value is not None:

            stats_path = Path(
                stats_path_value
            )

        else:

            # Old seed 2026 fallback
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
    # VALIDATION SCORES
    # ========================================================

    validation_scores = (
        load_validation_scores(
            validation_path
        )
    )

    validation_eer, reconstructed_threshold = (
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

        fixed_threshold = float(
            reconstructed_threshold
        )

    # ========================================================
    # BEST EPOCH
    # ========================================================

    best_epoch = checkpoint.get(
        "epoch",
        checkpoint.get(
            "best_epoch",
            -1
        )
    )

    return (
        model,
        checkpoint,
        stats,
        validation_scores,
        validation_eer,
        fixed_threshold,
        int(best_epoch)
    )


# ============================================================
# SIMULATED UNSEEN MANIFEST
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
        "rir_group"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Simulated manifest kehilangan "
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

    if len(
        test
    ) != 200:

        raise RuntimeError(
            "Simulated unseen seharusnya "
            f"200 rows, ditemukan {len(test)}."
        )

    label_counts = (
        test[
            "label"
        ]
        .value_counts()
        .to_dict()
    )

    if (
        label_counts.get(
            0,
            0
        )
        !=
        100
        or
        label_counts.get(
            1,
            0
        )
        !=
        100
    ):

        raise RuntimeError(
            "Simulated unseen harus "
            "100 bona fide dan 100 spoof."
        )

    return test.reset_index(
        drop=True
    )


# ============================================================
# REAL REPLAY MANIFEST
# ============================================================

def load_real_replay():

    if not REAL_REPLAY_MANIFEST.exists():

        raise FileNotFoundError(
            REAL_REPLAY_MANIFEST
        )

    df = pd.read_csv(
        REAL_REPLAY_MANIFEST
    )

    path_column = None

    for candidate in [
        "recording_path",
        "file_path",
        "output_path"
    ]:

        if candidate in df.columns:

            path_column = candidate
            break

    if path_column is None:

        raise RuntimeError(
            "Kolom audio path real replay "
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
            "Real replay manifest kehilangan "
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

    if len(
        df
    ) != 480:

        raise RuntimeError(
            "Real replay seharusnya "
            f"480 rows, ditemukan {len(df)}."
        )

    return df.reset_index(
        drop=True
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def evaluate_dataframe(
    model,
    dataframe,
    audio_path_column,
    extractor,
    stats,
    feature_set,
    device,
    model_name,
    use_grl,
    seed,
    condition,
    fixed_threshold
):

    rows = []

    total = len(
        dataframe
    )

    for index, row in (
        dataframe.iterrows()
    ):

        wav = load_audio_4s(
            row[
                audio_path_column
            ]
        )

        features = extract_features(
            wav,
            extractor,
            stats,
            feature_set
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

        output = {

            "file_id":
                row.get(
                    "file_id",
                    row.get(
                        "recording_id",
                        ""
                    )
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

            "condition":
                condition,

            "seed":
                int(seed),

            "model":
                model_name,

            "feature_set":
                feature_set,

            "use_grl":
                int(
                    use_grl
                ),

            "fixed_threshold":
                fixed_threshold
        }

        optional_columns = [

            "rir_identity",
            "rir_group",
            "snr_db",

            "recording_id",

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
# SUMMARY ROW
# ============================================================

def build_run_row(
    model_name,
    feature_set,
    use_grl,
    seed,
    best_epoch,
    validation_eer,
    condition,
    scores_df,
    fixed_threshold
):

    metrics = compute_metrics(

        scores_df[
            "label"
        ].values,

        scores_df[
            "fake_score"
        ].values,

        fixed_threshold
    )

    return {

        "model":
            model_name,

        "feature_set":
            feature_set,

        "use_grl":
            int(
                use_grl
            ),

        "seed":
            int(seed),

        "best_epoch":
            int(best_epoch),

        "validation_eer":
            float(
                validation_eer
            ),

        "condition":
            condition,

        "n":
            int(
                len(
                    scores_df
                )
            ),

        "test_eer":
            float(
                metrics[
                    "eer"
                ]
            ),

        "auc":
            float(
                metrics[
                    "auc"
                ]
            ),

        "f1":
            float(
                metrics[
                    "f1"
                ]
            ),

        "balanced_accuracy":
            float(
                metrics[
                    "balanced_accuracy"
                ]
            ),

        "accuracy":
            float(
                metrics[
                    "accuracy"
                ]
            ),

        "validation_threshold":
            float(
                fixed_threshold
            ),

        "test_eer_threshold":
            float(
                metrics[
                    "eer_threshold"
                ]
            )
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
        "TAHAP 8 - MULTI-SEED EVALUATION"
    )
    print(
        "=============================================="
    )

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

    print(
        "Seeds:",
        SEEDS
    )

    print(
        "Total experiments:",
        len(
            EXPERIMENTS
        )
    )

    # ========================================================
    # LOAD DATASETS
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

    print()
    print(
        "Real microphone distribution:"
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

    extractor = SpectralPhase(
        sr=TARGET_SR,
        n_fft=512,
        hop=160,
        n_mels=80
    )

    extractor.eval()

    # ========================================================
    # OUTPUT ACCUMULATORS
    # ========================================================

    all_scores = []
    run_rows = []

    # ========================================================
    # LOOP ALL 12 RUNS
    # ========================================================

    for experiment_index, experiment in enumerate(
        EXPERIMENTS,
        start=1
    ):

        model_name = (
            experiment[
                "model_name"
            ]
        )

        feature_set = (
            experiment[
                "feature_set"
            ]
        )

        use_grl = bool(
            experiment[
                "use_grl"
            ]
        )

        seed = int(
            experiment[
                "seed"
            ]
        )

        print()
        print(
            "=============================================="
        )

        print(
            f"RUN {experiment_index}/"
            f"{len(EXPERIMENTS)}"
        )

        print(
            f"{model_name} | Seed {seed}"
        )

        print(
            "=============================================="
        )

        (
            model,
            checkpoint,
            stats,
            validation_scores,
            validation_eer,
            fixed_threshold,
            best_epoch
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
            "GRL:",
            "ON"
            if use_grl
            else "OFF"
        )

        print(
            "Seed:",
            seed
        )

        print(
            "Best epoch:",
            best_epoch
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

        validation_output = (
            validation_scores[
                [
                    "label",
                    "fake_score"
                ]
            ]
            .copy()
        )

        # Add source id if available
        if (
            "source_file_id"
            in validation_scores.columns
        ):

            validation_output[
                "source_file_id"
            ] = (
                validation_scores[
                    "source_file_id"
                ]
                .astype(str)
            )

        else:

            validation_output[
                "source_file_id"
            ] = ""

        if (
            "file_id"
            in validation_scores.columns
        ):

            validation_output[
                "file_id"
            ] = validation_scores[
                "file_id"
            ]

        else:

            validation_output[
                "file_id"
            ] = ""

        validation_output[
            "condition"
        ] = "validation_seen"

        validation_output[
            "seed"
        ] = seed

        validation_output[
            "model"
        ] = model_name

        validation_output[
            "feature_set"
        ] = feature_set

        validation_output[
            "use_grl"
        ] = int(
            use_grl
        )

        validation_output[
            "fixed_threshold"
        ] = fixed_threshold

        all_scores.append(
            validation_output
        )

        run_rows.append(
            build_run_row(

                model_name=
                    model_name,

                feature_set=
                    feature_set,

                use_grl=
                    use_grl,

                seed=
                    seed,

                best_epoch=
                    best_epoch,

                validation_eer=
                    validation_eer,

                condition=
                    "validation_seen",

                scores_df=
                    validation_output,

                fixed_threshold=
                    fixed_threshold
            )
        )

        # ====================================================
        # SIMULATED UNSEEN
        # ====================================================

        print()
        print(
            "Simulated unseen RIR:"
        )

        sim_scores = evaluate_dataframe(

            model=
                model,

            dataframe=
                sim_unseen,

            audio_path_column=
                "file_path",

            extractor=
                extractor,

            stats=
                stats,

            feature_set=
                feature_set,

            device=
                device,

            model_name=
                model_name,

            use_grl=
                use_grl,

            seed=
                seed,

            condition=
                "simulated_unseen_rir",

            fixed_threshold=
                fixed_threshold
        )

        all_scores.append(
            sim_scores
        )

        run_rows.append(
            build_run_row(

                model_name=
                    model_name,

                feature_set=
                    feature_set,

                use_grl=
                    use_grl,

                seed=
                    seed,

                best_epoch=
                    best_epoch,

                validation_eer=
                    validation_eer,

                condition=
                    "simulated_unseen_rir",

                scores_df=
                    sim_scores,

                fixed_threshold=
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

        real_scores = evaluate_dataframe(

            model=
                model,

            dataframe=
                real_replay,

            audio_path_column=
                "_audio_path",

            extractor=
                extractor,

            stats=
                stats,

            feature_set=
                feature_set,

            device=
                device,

            model_name=
                model_name,

            use_grl=
                use_grl,

            seed=
                seed,

            condition=
                "real_replay",

            fixed_threshold=
                fixed_threshold
        )

        all_scores.append(
            real_scores
        )

        run_rows.append(
            build_run_row(

                model_name=
                    model_name,

                feature_set=
                    feature_set,

                use_grl=
                    use_grl,

                seed=
                    seed,

                best_epoch=
                    best_epoch,

                validation_eer=
                    validation_eer,

                condition=
                    "real_replay",

                scores_df=
                    real_scores,

                fixed_threshold=
                    fixed_threshold
            )
        )

        # ====================================================
        # MIC UNSEEN
        # ====================================================

        mic_unseen_scores = real_scores[
            real_scores[
                "mic"
            ]
            .astype(str)
            .str.strip()
            ==
            "mic_unseen"
        ].copy()

        if len(
            mic_unseen_scores
        ) != 240:

            raise RuntimeError(
                f"{model_name} seed {seed}: "
                "mic_unseen seharusnya 240, "
                f"ditemukan "
                f"{len(mic_unseen_scores)}."
            )

        # Give explicit condition
        mic_unseen_scores[
            "condition"
        ] = "real_replay_mic_unseen"

        all_scores.append(
            mic_unseen_scores
        )

        run_rows.append(
            build_run_row(

                model_name=
                    model_name,

                feature_set=
                    feature_set,

                use_grl=
                    use_grl,

                seed=
                    seed,

                best_epoch=
                    best_epoch,

                validation_eer=
                    validation_eer,

                condition=
                    "real_replay_mic_unseen",

                scores_df=
                    mic_unseen_scores,

                fixed_threshold=
                    fixed_threshold
            )
        )

    # ========================================================
    # CONCAT SCORES
    # ========================================================

    scores_df = pd.concat(
        all_scores,
        ignore_index=True,
        sort=False
    )

    run_summary = pd.DataFrame(
        run_rows
    )

    # ========================================================
    # CONDITION SUMMARY
    # Mean ± standard deviation across 3 seeds
    # ========================================================

    metric_columns = [

        "test_eer",
        "auc",
        "f1",
        "balanced_accuracy",
        "accuracy"
    ]

    grouped_rows = []

    group_columns = [
        "model",
        "feature_set",
        "use_grl",
        "condition"
    ]

    for group_key, group_df in (
        run_summary
        .groupby(
            group_columns,
            sort=False
        )
    ):

        (
            model_name,
            feature_set,
            use_grl,
            condition
        ) = group_key

        row = {

            "model":
                model_name,

            "feature_set":
                feature_set,

            "use_grl":
                int(
                    use_grl
                ),

            "condition":
                condition,

            "n_seeds":
                int(
                    group_df[
                        "seed"
                    ]
                    .nunique()
                )
        }

        for metric in metric_columns:

            values = (
                group_df[
                    metric
                ]
                .astype(float)
                .values
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(
                    values
                )
            )

            row[
                f"{metric}_std"
            ] = float(
                np.std(
                    values,
                    ddof=1
                )
            )

        grouped_rows.append(
            row
        )

    condition_summary = pd.DataFrame(
        grouped_rows
    )

    # ========================================================
    # CHECK THAT EVERY MODEL HAS 3 SEEDS
    # ========================================================

    bad_seed_groups = (
        condition_summary[
            condition_summary[
                "n_seeds"
            ]
            !=
            3
        ]
    )

    if not bad_seed_groups.empty:

        print()
        print(
            "WARNING:"
        )

        print(
            "Ada grup kondisi yang tidak "
            "memiliki 3 seed:"
        )

        print(
            bad_seed_groups[
                [
                    "model",
                    "condition",
                    "n_seeds"
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # REPORT TABLE
    # EER mean ± std in percent
    # ========================================================

    report_source = (
        condition_summary[
            [
                "model",
                "condition",
                "test_eer_mean",
                "test_eer_std"
            ]
        ]
        .copy()
    )

    report_source[
        "eer_mean_percent"
    ] = (
        report_source[
            "test_eer_mean"
        ]
        *
        100.0
    )

    report_source[
        "eer_std_percent"
    ] = (
        report_source[
            "test_eer_std"
        ]
        *
        100.0
    )

    report_source[
        "eer_mean_std"
    ] = report_source.apply(

        lambda row:
            (
                f"{row['eer_mean_percent']:.2f}"
                f" ± "
                f"{row['eer_std_percent']:.2f}"
            ),

        axis=1
    )

    report_table = (
        report_source
        .pivot(
            index="model",
            columns="condition",
            values="eer_mean_std"
        )
        .reset_index()
    )

    report_table = report_table.rename(

        columns={

            "validation_seen":
                "Validation Seen EER (%)",

            "simulated_unseen_rir":
                "Sim Unseen RIR EER (%)",

            "real_replay":
                "Real Replay EER (%)",

            "real_replay_mic_unseen":
                "Unseen Mic EER (%)"
        }
    )

    # --------------------------------------------------------
    # Fixed model order
    # --------------------------------------------------------

    model_order = {

        "Log-Mel only":
            0,

        "Log-Mel + Phase":
            1,

        "Full No-GRL":
            2,

        "Full + GRL":
            3
    }

    report_table[
        "_order"
    ] = (
        report_table[
            "model"
        ]
        .map(
            model_order
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

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    scores_df.to_csv(
        OUTPUT_SCORES,
        index=False
    )

    run_summary.to_csv(
        OUTPUT_RUN_SUMMARY,
        index=False
    )

    condition_summary.to_csv(
        OUTPUT_CONDITION_SUMMARY,
        index=False
    )

    report_table.to_csv(
        OUTPUT_REPORT_TABLE,
        index=False
    )

    # ========================================================
    # CONFIG
    # ========================================================

    config = {

        "seeds":
            SEEDS,

        "n_experiments":
            len(
                EXPERIMENTS
            ),

        "conditions": [

            "validation_seen",
            "simulated_unseen_rir",
            "real_replay",
            "real_replay_mic_unseen"
        ],

        "simulated_unseen_manifest":
            str(
                SIM_UNSEEN_MANIFEST
            ),

        "real_replay_manifest":
            str(
                REAL_REPLAY_MANIFEST
            ),

        "outputs": {

            "scores":
                str(
                    OUTPUT_SCORES
                ),

            "run_summary":
                str(
                    OUTPUT_RUN_SUMMARY
                ),

            "condition_summary":
                str(
                    OUTPUT_CONDITION_SUMMARY
                ),

            "report_table":
                str(
                    OUTPUT_REPORT_TABLE
                )
        },

        "metric_policy": {

            "eer":
                (
                    "Calculated from continuous "
                    "fake scores for each "
                    "seed and condition."
                ),

            "auc":
                (
                    "Calculated from continuous "
                    "fake scores."
                ),

            "f1":
                (
                    "Calculated using frozen "
                    "validation EER threshold "
                    "of each seed."
                ),

            "balanced_accuracy":
                (
                    "Calculated using frozen "
                    "validation EER threshold "
                    "of each seed."
                ),

            "accuracy":
                (
                    "Calculated using frozen "
                    "validation EER threshold "
                    "of each seed."
                ),

            "multiseed":
                (
                    "Mean and sample standard "
                    "deviation across seeds "
                    "2026, 2027, 2028."
                )
        }
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
    # TERMINAL OUTPUT
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "RUN SUMMARY"
    )

    print(
        "=============================================="
    )

    print()

    print(
        run_summary[
            [
                "model",
                "seed",
                "best_epoch",
                "validation_eer",
                "condition",
                "test_eer",
                "auc",
                "balanced_accuracy"
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=============================================="
    )

    print(
        "MULTI-SEED CONDITION SUMMARY"
    )

    print(
        "=============================================="
    )

    print()

    for _, row in (
        condition_summary.iterrows()
    ):

        print(
            f"{row['model']:18s} | "
            f"{row['condition']:24s} | "
            f"EER="
            f"{row['test_eer_mean'] * 100:6.2f}% "
            f"± "
            f"{row['test_eer_std'] * 100:5.2f}% | "
            f"AUC="
            f"{row['auc_mean']:.4f} "
            f"± "
            f"{row['auc_std']:.4f}"
        )

    print()
    print(
        "=============================================="
    )

    print(
        "FINAL REPORT TABLE"
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

    print()
    print(
        "=============================================="
    )

    print(
        "MULTI-SEED EVALUATION SELESAI"
    )

    print(
        "=============================================="
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_SCORES
    )

    print(
        OUTPUT_RUN_SUMMARY
    )

    print(
        OUTPUT_CONDITION_SUMMARY
    )

    print(
        OUTPUT_REPORT_TABLE
    )

    print(
        OUTPUT_CONFIG
    )

if __name__ == "__main__":
    main()