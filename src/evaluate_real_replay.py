from pathlib import Path
import json

import numpy as np
import pandas as pd
import soundfile as sf

import torch
from scipy.signal import resample_poly
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score
)
from torch.utils.data import Dataset, DataLoader

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

REAL_MANIFEST = Path(
    "manifests/real_replay_manifest_480.csv"
)

TRAIN_MANIFEST = Path(
    "manifests/train_replay_manifest.csv"
)

NO_GRL_CHECKPOINT = Path(
    "checkpoints/replay_no_grl.pt"
)

WITH_GRL_CHECKPOINT = Path(
    "checkpoints/replay_with_grl.pt"
)

NO_GRL_VAL_SCORES = Path(
    "results/scores_no_grl_validation.csv"
)

WITH_GRL_VAL_SCORES = Path(
    "results/scores_with_grl_validation.csv"
)

STATS_PATH = Path(
    "data/processed/three_channel_stats.pt"
)

RESULT_DIR = Path(
    "results"
)

N_DOMAIN = 5
EMBEDDING_DIM = 256
BATCH_SIZE = 8


# ============================================================
# HELPERS
# ============================================================

def find_first_column(
    df,
    candidates,
    required=True
):

    for column in candidates:

        if column in df.columns:
            return column

    if required:

        raise ValueError(
            "Tidak menemukan salah satu kolom berikut: "
            f"{candidates}\n"
            f"Kolom yang tersedia:\n"
            f"{df.columns.tolist()}"
        )

    return None


def normalize_string(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


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
    # MULTICHANNEL -> MONO
    # --------------------------------------------------------

    audio = audio.mean(
        axis=1
    )

    # --------------------------------------------------------
    # RESAMPLE 48 kHz -> 16 kHz
    # bila diperlukan
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
    # PAD
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
    # CENTER CROP
    # --------------------------------------------------------

    elif n > target_length:

        start = (
            n - target_length
        ) // 2

        audio = audio[
            start:
            start + target_length
        ]

    # --------------------------------------------------------
    # PEAK NORMALIZATION
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

    wav = torch.from_numpy(
        audio
    ).float()

    # [1, T]
    wav = wav.unsqueeze(0)

    return wav


# ============================================================
# EER
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
        1.0
        -
        tpr
    )

    index = np.nanargmin(
        np.abs(
            fpr
            -
            fnr
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

    threshold = float(
        thresholds[index]
    )

    return (
        eer,
        threshold
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def compute_metrics(
    y_true,
    scores,
    fixed_threshold
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(
        np.unique(
            y_true
        )
    ) < 2:

        raise ValueError(
            "EER/AUC membutuhkan "
            "kedua kelas label 0 dan 1."
        )

    eer, test_eer_threshold = (
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

        "eer":
            eer,

        "test_eer_threshold":
            test_eer_threshold,

        "auc":
            auc,

        "f1":
            f1,

        "balanced_accuracy":
            balanced_accuracy,

        "accuracy":
            accuracy,

        "fixed_threshold":
            float(
                fixed_threshold
            )
    }


# ============================================================
# GET VALIDATION THRESHOLD
# ============================================================

def get_validation_eer_threshold(
    score_path
):

    if not score_path.exists():

        raise FileNotFoundError(
            f"Validation score tidak ditemukan: "
            f"{score_path}"
        )

    df = pd.read_csv(
        score_path
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

        raise ValueError(
            f"{score_path} kehilangan "
            f"kolom {missing}"
        )

    eer, threshold = compute_eer(
        df[
            "label"
        ].values,
        df[
            "fake_score"
        ].values
    )

    return (
        eer,
        threshold
    )


# ============================================================
# DATASET
# ============================================================

class RealReplayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        feature_extractor,
        stats,
        path_column
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.feature_extractor = (
            feature_extractor
        )

        self.stats = stats

        self.path_column = (
            path_column
        )

    def __len__(self):

        return len(
            self.df
        )

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        wav = load_audio_4s(
            row[
                self.path_column
            ]
        )

        # [1,T] -> [B,1,T]

        wav = wav.unsqueeze(0)

        with torch.no_grad():

            logmel, phase = (
                self.feature_extractor(
                    wav
                )
            )

            mgd = modified_group_delay(
                wav,
                self.feature_extractor.fb
            )

        # [1,80,397] -> [80,397]

        logmel = logmel.squeeze(0)
        phase = phase.squeeze(0)
        mgd = mgd.squeeze(0)

        # ----------------------------------------------------
        # FINITE CHECK
        # ----------------------------------------------------

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
                    f"{name} memiliki NaN/Inf "
                    f"pada index {index}"
                )

        # ----------------------------------------------------
        # TRAINING-ONLY STANDARDIZATION
        # ----------------------------------------------------

        logmel = (
            logmel
            -
            self.stats[
                "logmel_mean"
            ]
        ) / (
            self.stats[
                "logmel_std"
            ]
            +
            1e-6
        )

        phase = (
            phase
            -
            self.stats[
                "phase_mean"
            ]
        ) / (
            self.stats[
                "phase_std"
            ]
            +
            1e-6
        )

        mgd = (
            mgd
            -
            self.stats[
                "mgd_mean"
            ]
        ) / (
            self.stats[
                "mgd_std"
            ]
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

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        def get_value(
            column,
            default=""
        ):

            if column in row.index:

                value = row[
                    column
                ]

                if pd.isna(
                    value
                ):
                    return default

                return str(
                    value
                )

            return default

        return {

            "features":
                features.float(),

            "label":
                torch.tensor(
                    int(
                        row[
                            "label"
                        ]
                    ),
                    dtype=torch.long
                ),

            "recording_id":
                get_value(
                    "recording_id",
                    str(index)
                ),

            "source_file_id":
                get_value(
                    "source_file_id",
                    ""
                ),

            "room":
                get_value(
                    "room",
                    ""
                ),

            "mic":
                get_value(
                    "mic",
                    ""
                ),

            "playback":
                get_value(
                    "playback",
                    ""
                ),

            "distance":
                get_value(
                    "distance",
                    ""
                ),

            "angle":
                get_value(
                    "angle",
                    ""
                ),

            "repetition":
                get_value(
                    "repetition",
                    ""
                )
        }


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path,
    device
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

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

    return (
        model,
        checkpoint
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    model_name,
    fixed_threshold
):

    model.eval()

    labels = []
    fake_scores = []

    rows = []

    total = len(
        loader.dataset
    )

    processed = 0

    print()
    print(
        "=============================================="
    )

    print(
        f"EVALUATING REAL REPLAY: {model_name}"
    )

    print(
        "=============================================="
    )

    print(
        "Fixed threshold dari seen validation:",
        f"{fixed_threshold:.8f}"
    )

    for batch in loader:

        x = batch[
            "features"
        ].to(device)

        y = batch[
            "label"
        ].to(device)

        (
            class_logits,
            transform_logits,
            _,
            _
        ) = model(
            x,
            grl_strength=0.0
        )

        probabilities = torch.softmax(
            class_logits,
            dim=1
        )

        fake_probability = (
            probabilities[
                :,
                1
            ]
            .detach()
            .cpu()
            .numpy()
        )

        y_numpy = (
            y
            .detach()
            .cpu()
            .numpy()
        )

        transform_prediction = (
            transform_logits
            .argmax(
                dim=1
            )
            .detach()
            .cpu()
            .numpy()
        )

        labels.extend(
            y_numpy.tolist()
        )

        fake_scores.extend(
            fake_probability.tolist()
        )

        batch_size = (
            y.size(0)
        )

        for i in range(
            batch_size
        ):

            predicted_label = int(
                fake_probability[i]
                >=
                fixed_threshold
            )

            rows.append({

                "model":
                    model_name,

                "condition":
                    "controlled_real_replay",

                "recording_id":
                    batch[
                        "recording_id"
                    ][i],

                "source_file_id":
                    batch[
                        "source_file_id"
                    ][i],

                "label":
                    int(
                        y_numpy[i]
                    ),

                "fake_score":
                    float(
                        fake_probability[i]
                    ),

                "fixed_threshold":
                    float(
                        fixed_threshold
                    ),

                "prediction":
                    predicted_label,

                "transform_prediction":
                    int(
                        transform_prediction[i]
                    ),

                "room":
                    batch[
                        "room"
                    ][i],

                "mic":
                    batch[
                        "mic"
                    ][i],

                "playback":
                    batch[
                        "playback"
                    ][i],

                "distance":
                    batch[
                        "distance"
                    ][i],

                "angle":
                    batch[
                        "angle"
                    ][i],

                "repetition":
                    batch[
                        "repetition"
                    ][i]
            })

        processed += (
            batch_size
        )

        if (
            processed % 40 == 0
            or
            processed == total
        ):

            print(
                f"[{processed}/{total}]"
            )

    metrics = compute_metrics(
        y_true=
            labels,
        scores=
            fake_scores,
        fixed_threshold=
            fixed_threshold
    )

    return (
        metrics,
        rows
    )


# ============================================================
# CONDITION TABLE
# ============================================================

def build_condition_table(
    score_df,
    thresholds
):

    rows = []

    condition_columns = [
        "room",
        "mic",
        "playback",
        "distance",
        "angle",
        "repetition"
    ]

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    for model_name in sorted(
        score_df[
            "model"
        ].unique()
    ):

        model_df = score_df[
            score_df[
                "model"
            ]
            ==
            model_name
        ]

        threshold = (
            thresholds[
                model_name
            ]
        )

        metrics = compute_metrics(
            model_df[
                "label"
            ].values,
            model_df[
                "fake_score"
            ].values,
            threshold
        )

        rows.append({

            "model":
                model_name,

            "condition_type":
                "overall",

            "condition_value":
                "controlled_real_replay",

            "n":
                len(
                    model_df
                ),

            "label_0":
                int(
                    (
                        model_df[
                            "label"
                        ]
                        ==
                        0
                    ).sum()
                ),

            "label_1":
                int(
                    (
                        model_df[
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

    # --------------------------------------------------------
    # PER CONDITION
    # --------------------------------------------------------

    for condition_column in condition_columns:

        if condition_column not in score_df.columns:
            continue

        nonempty = score_df[
            condition_column
        ].astype(str).str.strip()

        if (
            nonempty
            ==
            ""
        ).all():

            continue

        for model_name in sorted(
            score_df[
                "model"
            ].unique()
        ):

            model_df = score_df[
                score_df[
                    "model"
                ]
                ==
                model_name
            ]

            threshold = (
                thresholds[
                    model_name
                ]
            )

            for value, group in (
                model_df.groupby(
                    condition_column
                )
            ):

                # --------------------------------------------
                # EER/AUC only valid with both classes
                # --------------------------------------------

                if (
                    group[
                        "label"
                    ].nunique()
                    <
                    2
                ):

                    continue

                metrics = compute_metrics(
                    group[
                        "label"
                    ].values,
                    group[
                        "fake_score"
                    ].values,
                    threshold
                )

                rows.append({

                    "model":
                        model_name,

                    "condition_type":
                        condition_column,

                    "condition_value":
                        str(
                            value
                        ),

                    "n":
                        len(
                            group
                        ),

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

    return pd.DataFrame(
        rows
    )


# ============================================================
# MANIFEST AUDIT
# ============================================================

def audit_manifest(
    real
):

    print()
    print(
        "=============================================="
    )

    print(
        "REAL REPLAY MANIFEST AUDIT"
    )

    print(
        "=============================================="
    )

    print(
        "Rows:",
        len(
            real
        )
    )

    if len(
        real
    ) != 480:

        raise RuntimeError(
            "Real replay manifest seharusnya "
            f"480 rows, ditemukan {len(real)}."
        )

    print()
    print(
        "Label distribution:"
    )

    print(
        real[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    label_counts = (
        real[
            "label"
        ]
        .value_counts()
    )

    if (
        int(
            label_counts.get(
                0,
                0
            )
        )
        !=
        240
    ):

        raise RuntimeError(
            "Label 0 seharusnya 240."
        )

    if (
        int(
            label_counts.get(
                1,
                0
            )
        )
        !=
        240
    ):

        raise RuntimeError(
            "Label 1 seharusnya 240."
        )

    # --------------------------------------------------------
    # UNIQUE RECORDING
    # --------------------------------------------------------

    if "recording_id" in real.columns:

        duplicate_recording = int(
            real[
                "recording_id"
            ]
            .duplicated()
            .sum()
        )

        print()
        print(
            "Duplicate recording_id:",
            duplicate_recording
        )

        if duplicate_recording > 0:

            raise RuntimeError(
                "Duplicate recording_id ditemukan."
            )

    # --------------------------------------------------------
    # UNIQUE SOURCE
    # --------------------------------------------------------

    if "source_file_id" in real.columns:

        unique_source = int(
            real[
                "source_file_id"
            ]
            .nunique()
        )

        print(
            "Unique source:",
            unique_source
        )

        if unique_source != 120:

            print(
                "PERINGATAN: target sebelumnya "
                "adalah 120 unique source."
            )

        rows_per_source = (
            real
            .groupby(
                "source_file_id"
            )
            .size()
        )

        print()
        print(
            "Rows per source:"
        )

        print(
            rows_per_source
            .value_counts()
            .sort_index()
        )

    # --------------------------------------------------------
    # AVAILABLE CONDITION COLUMNS
    # --------------------------------------------------------

    condition_columns = [
        "room",
        "mic",
        "playback",
        "distance",
        "angle",
        "repetition"
    ]

    for column in condition_columns:

        if column in real.columns:

            print()
            print(
                f"{column}:"
            )

            print(
                real[
                    column
                ]
                .value_counts(
                    dropna=False
                )
            )


# ============================================================
# MAIN
# ============================================================

def main():

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
        "=============================================="
    )

    print(
        "TAHAP 8 - CONTROLLED REAL REPLAY EVALUATION"
    )

    print(
        "=============================================="
    )

    print(
        "Device:",
        device
    )

    # ========================================================
    # CHECK REQUIRED FILES
    # ========================================================

    required_files = [

        REAL_MANIFEST,
        NO_GRL_CHECKPOINT,
        WITH_GRL_CHECKPOINT,
        NO_GRL_VAL_SCORES,
        WITH_GRL_VAL_SCORES,
        STATS_PATH
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Tidak ditemukan: {path}"
            )

    # ========================================================
    # LOAD REAL MANIFEST
    # ========================================================

    real = pd.read_csv(
        REAL_MANIFEST
    )

    print()
    print(
        "Manifest columns:"
    )

    print(
        real.columns.tolist()
    )

    # ========================================================
    # PATH COLUMN
    # ========================================================

    path_column = find_first_column(
        real,
        [
            "recording_path",
            "file_path",
            "output_path"
        ]
    )

    print()
    print(
        "Audio path column:",
        path_column
    )

    # ========================================================
    # LABEL
    # ========================================================

    if "label" not in real.columns:

        raise ValueError(
            "real replay manifest tidak "
            "memiliki kolom label."
        )

    real[
        "label"
    ] = pd.to_numeric(
        real[
            "label"
        ],
        errors="raise"
    ).astype(int)

    if not real[
        "label"
    ].isin(
        [
            0,
            1
        ]
    ).all():

        raise RuntimeError(
            "Label real replay harus 0/1."
        )

    # ========================================================
    # OPTIONAL QC FILTER
    # ========================================================

    qc_column = find_first_column(
        real,
        [
            "qc_status",
            "qc",
            "status_qc"
        ],
        required=False
    )

    if qc_column is not None:

        real[
            qc_column
        ] = (
            normalize_string(
                real[
                    qc_column
                ]
            )
            .str.upper()
        )

        pass_rows = real[
            real[
                qc_column
            ]
            ==
            "PASS"
        ].copy()

        print()
        print(
            "QC PASS rows:",
            len(
                pass_rows
            )
        )

        # Final manifest Anda sebelumnya memang 480 PASS.
        if len(
            pass_rows
        ) == 480:

            real = pass_rows

        elif len(
            pass_rows
        ) > 0:

            raise RuntimeError(
                "Tidak semua 480 recording "
                "berstatus QC PASS."
            )

    # ========================================================
    # AUDIT MANIFEST
    # ========================================================

    audit_manifest(
        real
    )

    # ========================================================
    # CHECK SOURCE AGAINST TRAIN
    # ========================================================
    #
    # Ini bukan leakage dalam arti derivative split,
    # karena controlled replay memang dapat berasal dari
    # source yang dipilih untuk evaluasi fisik.
    #
    # Kita hanya melaporkan overlap, tidak otomatis gagal.
    # ========================================================

    if (
        TRAIN_MANIFEST.exists()
        and
        "source_file_id" in real.columns
    ):

        train_manifest = pd.read_csv(
            TRAIN_MANIFEST
        )

        if "source_file_id" in train_manifest.columns:

            train_sources = set(
                train_manifest[
                    "source_file_id"
                ]
                .astype(str)
            )

            real_sources = set(
                real[
                    "source_file_id"
                ]
                .astype(str)
            )

            overlap = (
                train_sources
                &
                real_sources
            )

            print()
            print(
                "Real source overlap dengan "
                "train manifest:",
                len(overlap)
            )

            print(
                "Catatan: angka ini dilaporkan "
                "untuk audit source identity."
            )

    # ========================================================
    # VALIDATE AUDIO PATHS
    # ========================================================

    print()
    print(
        "Checking audio paths..."
    )

    missing_audio = []

    for _, row in real.iterrows():

        path = Path(
            str(
                row[
                    path_column
                ]
            )
        )

        if not path.exists():

            missing_audio.append(
                str(
                    path
                )
            )

    print(
        "Missing audio:",
        len(
            missing_audio
        )
    )

    if missing_audio:

        print()
        print(
            "Contoh missing path:"
        )

        for path in missing_audio[
            :10
        ]:

            print(
                path
            )

        raise FileNotFoundError(
            f"Ada {len(missing_audio)} "
            "real replay audio tidak ditemukan."
        )

    # ========================================================
    # LOAD TRAINING-ONLY STATS
    # ========================================================

    stats = torch.load(
        STATS_PATH,
        map_location="cpu",
        weights_only=False
    )

    print()
    print(
        "Training statistics:"
    )

    print(
        "Log-Mel mean/std:",
        stats[
            "logmel_mean"
        ],
        stats[
            "logmel_std"
        ]
    )

    print(
        "Phase mean/std:",
        stats[
            "phase_mean"
        ],
        stats[
            "phase_std"
        ]
    )

    print(
        "MGD mean/std:",
        stats[
            "mgd_mean"
        ],
        stats[
            "mgd_std"
        ]
    )

    # ========================================================
    # VALIDATION THRESHOLDS
    # ========================================================

    (
        no_grl_validation_eer,
        no_grl_threshold
    ) = get_validation_eer_threshold(
        NO_GRL_VAL_SCORES
    )

    (
        with_grl_validation_eer,
        with_grl_threshold
    ) = get_validation_eer_threshold(
        WITH_GRL_VAL_SCORES
    )

    print()
    print(
        "Validation thresholds:"
    )

    print(
        "NO GRL :",
        f"EER={no_grl_validation_eer * 100:.2f}%",
        f"threshold={no_grl_threshold:.8f}"
    )

    print(
        "WITH GRL:",
        f"EER={with_grl_validation_eer * 100:.2f}%",
        f"threshold={with_grl_threshold:.8f}"
    )

    # ========================================================
    # FEATURE EXTRACTOR
    # ========================================================

    feature_extractor = SpectralPhase(
        sr=16000,
        n_fft=512,
        hop=160,
        n_mels=80
    )

    feature_extractor.eval()

    # ========================================================
    # DATASET / DATALOADER
    # ========================================================

    dataset = RealReplayDataset(

        dataframe=
            real,

        feature_extractor=
            feature_extractor,

        stats=
            stats,

        path_column=
            path_column
    )

    loader = DataLoader(

        dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=False
    )

    # ========================================================
    # LOAD CHECKPOINTS
    # ========================================================

    (
        no_grl_model,
        no_grl_checkpoint
    ) = load_model(
        NO_GRL_CHECKPOINT,
        device
    )

    (
        with_grl_model,
        with_grl_checkpoint
    ) = load_model(
        WITH_GRL_CHECKPOINT,
        device
    )

    print()
    print(
        "NO GRL checkpoint:"
    )

    print(
        "Epoch:",
        no_grl_checkpoint[
            "epoch"
        ]
    )

    print(
        "Seen validation EER:",
        f"{no_grl_checkpoint['validation_eer'] * 100:.2f}%"
    )

    print()
    print(
        "WITH GRL checkpoint:"
    )

    print(
        "Epoch:",
        with_grl_checkpoint[
            "epoch"
        ]
    )

    print(
        "Seen validation EER:",
        f"{with_grl_checkpoint['validation_eer'] * 100:.2f}%"
    )

    # ========================================================
    # NO GRL
    # ========================================================

    (
        no_grl_metrics,
        no_grl_rows
    ) = evaluate_model(

        model=
            no_grl_model,

        loader=
            loader,

        device=
            device,

        model_name=
            "no_grl",

        fixed_threshold=
            no_grl_threshold
    )

    # ========================================================
    # WITH GRL
    # ========================================================

    (
        with_grl_metrics,
        with_grl_rows
    ) = evaluate_model(

        model=
            with_grl_model,

        loader=
            loader,

        device=
            device,

        model_name=
            "with_grl",

        fixed_threshold=
            with_grl_threshold
    )

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SCORES
    # ========================================================

    no_grl_score_path = (
        RESULT_DIR
        /
        "scores_real_replay_no_grl.csv"
    )

    with_grl_score_path = (
        RESULT_DIR
        /
        "scores_real_replay_with_grl.csv"
    )

    combined_score_path = (
        RESULT_DIR
        /
        "scores_real_replay.csv"
    )

    no_grl_df = pd.DataFrame(
        no_grl_rows
    )

    with_grl_df = pd.DataFrame(
        with_grl_rows
    )

    combined_df = pd.concat(
        [
            no_grl_df,
            with_grl_df
        ],
        ignore_index=True
    )

    no_grl_df.to_csv(
        no_grl_score_path,
        index=False
    )

    with_grl_df.to_csv(
        with_grl_score_path,
        index=False
    )

    combined_df.to_csv(
        combined_score_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame([

        {

            "model":
                "no_grl",

            "checkpoint_epoch":
                no_grl_checkpoint[
                    "epoch"
                ],

            "seen_validation_eer":
                no_grl_checkpoint[
                    "validation_eer"
                ],

            "validation_threshold":
                no_grl_threshold,

            "real_replay_eer":
                no_grl_metrics[
                    "eer"
                ],

            "real_replay_auc":
                no_grl_metrics[
                    "auc"
                ],

            "real_replay_f1":
                no_grl_metrics[
                    "f1"
                ],

            "real_replay_balanced_accuracy":
                no_grl_metrics[
                    "balanced_accuracy"
                ],

            "real_replay_accuracy":
                no_grl_metrics[
                    "accuracy"
                ]
        },

        {

            "model":
                "with_grl",

            "checkpoint_epoch":
                with_grl_checkpoint[
                    "epoch"
                ],

            "seen_validation_eer":
                with_grl_checkpoint[
                    "validation_eer"
                ],

            "validation_threshold":
                with_grl_threshold,

            "real_replay_eer":
                with_grl_metrics[
                    "eer"
                ],

            "real_replay_auc":
                with_grl_metrics[
                    "auc"
                ],

            "real_replay_f1":
                with_grl_metrics[
                    "f1"
                ],

            "real_replay_balanced_accuracy":
                with_grl_metrics[
                    "balanced_accuracy"
                ],

            "real_replay_accuracy":
                with_grl_metrics[
                    "accuracy"
                ]
        }
    ])

    summary_path = (
        RESULT_DIR
        /
        "evaluation_real_replay_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # CONDITION TABLE
    # ========================================================

    thresholds = {

        "no_grl":
            no_grl_threshold,

        "with_grl":
            with_grl_threshold
    }

    condition_table = build_condition_table(
        combined_df,
        thresholds
    )

    condition_path = (
        RESULT_DIR
        /
        "evaluation_real_replay_conditions.csv"
    )

    condition_table.to_csv(
        condition_path,
        index=False
    )

    # ========================================================
    # CONFIG
    # ========================================================

    config = {

        "evaluation":
            "controlled_real_replay",

        "manifest":
            str(
                REAL_MANIFEST
            ),

        "audio_path_column":
            path_column,

        "test_rows":
            len(
                real
            ),

        "label_0":
            int(
                (
                    real[
                        "label"
                    ]
                    ==
                    0
                ).sum()
            ),

        "label_1":
            int(
                (
                    real[
                        "label"
                    ]
                    ==
                    1
                ).sum()
            ),

        "no_grl_checkpoint":
            str(
                NO_GRL_CHECKPOINT
            ),

        "with_grl_checkpoint":
            str(
                WITH_GRL_CHECKPOINT
            ),

        "threshold_source":
            "seen_validation_eer_threshold",

        "training_stats":
            str(
                STATS_PATH
            ),

        "condition_columns":
            [
                column
                for column in [
                    "room",
                    "mic",
                    "playback",
                    "distance",
                    "angle",
                    "repetition"
                ]
                if column in real.columns
            ],

        "important_note":
            (
                "Condition-level results are descriptive "
                "until room/mic/playback metadata are confirmed "
                "to correspond to actual distinct physical factors."
            )
    }

    config_path = (
        RESULT_DIR
        /
        "evaluation_real_replay_config.json"
    )

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
    # FINAL RESULT
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "CONTROLLED REAL REPLAY EVALUATION SELESAI"
    )

    print(
        "=============================================="
    )

    print()
    print(
        "NO GRL"
    )

    print(
        "Seen Val EER :",
        f"{no_grl_checkpoint['validation_eer'] * 100:.2f}%"
    )

    print(
        "Real EER     :",
        f"{no_grl_metrics['eer'] * 100:.2f}%"
    )

    print(
        "Real AUC     :",
        f"{no_grl_metrics['auc']:.4f}"
    )

    print(
        "Real F1      :",
        f"{no_grl_metrics['f1']:.4f}"
    )

    print(
        "Real BAcc    :",
        f"{no_grl_metrics['balanced_accuracy']:.4f}"
    )

    print(
        "Real Accuracy:",
        f"{no_grl_metrics['accuracy']:.4f}"
    )

    print()
    print(
        "WITH GRL"
    )

    print(
        "Seen Val EER :",
        f"{with_grl_checkpoint['validation_eer'] * 100:.2f}%"
    )

    print(
        "Real EER     :",
        f"{with_grl_metrics['eer'] * 100:.2f}%"
    )

    print(
        "Real AUC     :",
        f"{with_grl_metrics['auc']:.4f}"
    )

    print(
        "Real F1      :",
        f"{with_grl_metrics['f1']:.4f}"
    )

    print(
        "Real BAcc    :",
        f"{with_grl_metrics['balanced_accuracy']:.4f}"
    )

    print(
        "Real Accuracy:",
        f"{with_grl_metrics['accuracy']:.4f}"
    )

    # ========================================================
    # SIMULATED-TO-REAL GAP
    # ========================================================

    no_grl_gap = (
        no_grl_metrics[
            "eer"
        ]
        -
        no_grl_checkpoint[
            "validation_eer"
        ]
    )

    with_grl_gap = (
        with_grl_metrics[
            "eer"
        ]
        -
        with_grl_checkpoint[
            "validation_eer"
        ]
    )

    print()
    print(
        "Simulated/seen validation -> real EER gap:"
    )

    print(
        "NO GRL :",
        f"{no_grl_gap * 100:+.2f}",
        "percentage points"
    )

    print(
        "WITH GRL:",
        f"{with_grl_gap * 100:+.2f}",
        "percentage points"
    )

    # ========================================================
    # GRL COMPARISON
    # ========================================================

    eer_difference = (
        no_grl_metrics[
            "eer"
        ]
        -
        with_grl_metrics[
            "eer"
        ]
    )

    print()
    print(
        "Real replay EER difference:"
    )

    print(
        "NO GRL - WITH GRL =",
        f"{eer_difference * 100:.2f}",
        "percentage points"
    )

    print()

    if (
        with_grl_metrics[
            "eer"
        ]
        <
        no_grl_metrics[
            "eer"
        ]
    ):

        print(
            "HASIL SEMENTARA:"
        )

        print(
            "WITH GRL memiliki EER lebih rendah "
            "pada controlled real replay."
        )

        print(
            "Hasil ini konsisten dengan peningkatan "
            "generalisasi simulated-to-real."
        )

    elif (
        with_grl_metrics[
            "eer"
        ]
        >
        no_grl_metrics[
            "eer"
        ]
    ):

        print(
            "HASIL SEMENTARA:"
        )

        print(
            "WITH GRL memiliki EER lebih tinggi "
            "pada controlled real replay."
        )

        print(
            "Pada eksperimen ini GRL belum "
            "meningkatkan generalisasi simulated-to-real."
        )

    else:

        print(
            "HASIL SEMENTARA:"
        )

        print(
            "NO GRL dan WITH GRL memiliki "
            "EER real replay yang sama."
        )

    print()
    print(
        "PENTING:"
    )

    print(
        "Hasil overall 480 boleh dianalisis sebagai "
        "controlled physical replay."
    )

    print(
        "Hasil per room/mic/playback belum boleh "
        "disebut unseen-factor evaluation sebelum "
        "metadata tersebut dipastikan mewakili "
        "kondisi fisik yang benar-benar berbeda."
    )

    print()
    print(
        "Output:"
    )

    print(
        no_grl_score_path
    )

    print(
        with_grl_score_path
    )

    print(
        combined_score_path
    )

    print(
        summary_path
    )

    print(
        condition_path
    )

    print(
        config_path
    )

if __name__ == "__main__":
    main()