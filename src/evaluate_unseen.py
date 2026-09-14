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

SIM_MANIFEST = Path(
    "manifests/manifest_replay_sim_week9.csv"
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


# ============================================================
# AUDIO LOADER
# ============================================================

def load_audio_4s(
    path,
    target_sr=TARGET_SR,
    seconds=SECONDS
):

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
        target_sr * seconds
    )

    n = len(audio)

    # ========================================================
    # PAD
    # ========================================================

    if n < target_length:

        audio = np.pad(
            audio,
            (
                0,
                target_length - n
            ),
            mode="constant"
        )

    # ========================================================
    # CENTER CROP
    # ========================================================

    elif n > target_length:

        start = (
            n - target_length
        ) // 2

        audio = audio[
            start:
            start + target_length
        ]

    # ========================================================
    # PEAK NORMALIZATION
    # ========================================================

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    if peak > 1e-6:

        audio = (
            audio / peak
        )

    wav = torch.from_numpy(
        audio
    ).float()

    # [1, T]
    wav = wav.unsqueeze(0)

    return wav


# ============================================================
# METRIC FUNCTIONS
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

    threshold = float(
        thresholds[index]
    )

    return (
        eer,
        threshold
    )


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

    accuracy = float(
        (
            prediction
            ==
            y_true
        ).mean()
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
# VALIDATION THRESHOLD
# ============================================================

def get_validation_eer_threshold(
    score_path
):

    if not score_path.exists():

        raise FileNotFoundError(
            f"Validation scores tidak ditemukan: "
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
            f"{score_path} kehilangan kolom: "
            f"{missing}"
        )

    eer, threshold = compute_eer(
        df["label"].values,
        df["fake_score"].values
    )

    return (
        eer,
        threshold
    )


# ============================================================
# DATASET
# ============================================================

class UnseenReplayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        feature_extractor,
        stats
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
            row["file_path"]
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

        # [1,80,T] -> [80,T]
        logmel = logmel.squeeze(0)
        phase = phase.squeeze(0)
        mgd = mgd.squeeze(0)

        # ====================================================
        # FINITE CHECK
        # ====================================================

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
                    f"{name} NaN/Inf pada "
                    f"{row['file_id']}"
                )

        # ====================================================
        # TRAINING-ONLY STANDARDIZATION
        # ====================================================

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

        result = {

            "features":
                features.float(),

            "label":
                torch.tensor(
                    int(
                        row["label"]
                    ),
                    dtype=torch.long
                ),

            "file_id":
                str(
                    row["file_id"]
                ),

            "source_file_id":
                str(
                    row["source_file_id"]
                ),

            "rir_identity":
                str(
                    row.get(
                        "rir_identity",
                        ""
                    )
                ),

            "rir_group":
                str(
                    row.get(
                        "rir_group",
                        ""
                    )
                ),

            "snr_db":
                str(
                    row.get(
                        "snr_db",
                        ""
                    )
                )
        }

        return result


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
# EVALUATE ONE MODEL
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
        f"EVALUATING: {model_name}"
    )

    print(
        "=============================================="
    )

    print(
        "Fixed threshold dari validation:",
        f"{fixed_threshold:.8f}"
    )

    for batch in loader:

        x = batch[
            "features"
        ].to(device)

        y = batch[
            "label"
        ].to(device)

        # ====================================================
        # FORWARD
        # ====================================================
        #
        # GRL tidak dibutuhkan saat inference.
        # Embedding/classification logits tidak berubah
        # karena GRL hanya memengaruhi backward domain branch.
        # ====================================================

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
                    "simulated_unseen",

                "file_id":
                    batch[
                        "file_id"
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

                "rir_identity":
                    batch[
                        "rir_identity"
                    ][i],

                "rir_group":
                    batch[
                        "rir_group"
                    ][i],

                "snr_db":
                    batch[
                        "snr_db"
                    ][i]
            })

        processed += (
            batch_size
        )

        if (
            processed % 50 == 0
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
# CONDITION ANALYSIS
# ============================================================

def build_condition_table(
    score_df,
    validation_thresholds
):

    results = []

    # ========================================================
    # OVERALL PER MODEL
    # ========================================================

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
            validation_thresholds[
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

        results.append({

            "model":
                model_name,

            "condition_type":
                "overall",

            "condition_value":
                "simulated_unseen",

            "n":
                len(
                    model_df
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

    # ========================================================
    # PER RIR IDENTITY
    # ========================================================

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
            validation_thresholds[
                model_name
            ]
        )

        for rir_identity, group in (
            model_df.groupby(
                "rir_identity"
            )
        ):

            # EER/AUC memerlukan dua kelas.
            if group[
                "label"
            ].nunique() < 2:

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

            results.append({

                "model":
                    model_name,

                "condition_type":
                    "rir_identity",

                "condition_value":
                    rir_identity,

                "n":
                    len(
                        group
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

    # ========================================================
    # PER SNR
    # ========================================================

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
            validation_thresholds[
                model_name
            ]
        )

        for snr, group in (
            model_df.groupby(
                "snr_db"
            )
        ):

            if group[
                "label"
            ].nunique() < 2:

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

            results.append({

                "model":
                    model_name,

                "condition_type":
                    "snr_db",

                "condition_value":
                    snr,

                "n":
                    len(
                        group
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
        results
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
        "TAHAP 8 - SIMULATED UNSEEN RIR EVALUATION"
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

        SIM_MANIFEST,
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
    # LOAD SIMULATED MANIFEST
    # ========================================================

    sim = pd.read_csv(
        SIM_MANIFEST
    )

    required_columns = [

        "file_id",
        "source_file_id",
        "file_path",
        "label",
        "split",
        "rir_group",
        "rir_identity",
        "snr_db",
        "qc_status"
    ]

    missing = [

        column

        for column in required_columns

        if column not in sim.columns
    ]

    if missing:

        raise ValueError(
            "Simulated manifest kehilangan "
            f"kolom: {missing}"
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    sim["split"] = (
        sim["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    sim["rir_group"] = (
        sim["rir_group"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    sim["qc_status"] = (
        sim["qc_status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    sim["label"] = pd.to_numeric(
        sim["label"],
        errors="raise"
    ).astype(int)

    # ========================================================
    # FILTER TEST UNSEEN
    # ========================================================

    unseen = sim[

        (
            sim[
                "split"
            ]
            ==
            "test"
        )

        &

        (
            sim[
                "rir_group"
            ]
            ==
            "unseen"
        )

        &

        (
            sim[
                "qc_status"
            ]
            ==
            "PASS"
        )

    ].copy()

    print()
    print(
        "Sim manifest rows:",
        len(sim)
    )

    print(
        "Unseen test rows:",
        len(unseen)
    )

    if len(
        unseen
    ) != 200:

        raise RuntimeError(
            "Simulated unseen test "
            "seharusnya 200, "
            f"ditemukan {len(unseen)}."
        )

    # ========================================================
    # LABEL BALANCE
    # ========================================================

    print()
    print(
        "Label distribution:"
    )

    print(
        unseen[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    label_counts = (
        unseen[
            "label"
        ]
        .value_counts()
    )

    if int(
        label_counts.get(
            0,
            0
        )
    ) != 100:

        raise RuntimeError(
            "Label 0 unseen harus 100."
        )

    if int(
        label_counts.get(
            1,
            0
        )
    ) != 100:

        raise RuntimeError(
            "Label 1 unseen harus 100."
        )

    # ========================================================
    # RIR DISTRIBUTION
    # ========================================================

    print()
    print(
        "Unseen RIR identities:"
    )

    print(
        unseen[
            "rir_identity"
        ]
        .value_counts()
    )

    print()
    print(
        "SNR distribution:"
    )

    print(
        unseen[
            "snr_db"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # SOURCE LEAKAGE CHECK
    # ========================================================
    #
    # Pastikan source unseen tidak muncul pada train/validation
    # di train_replay_manifest.csv bila source split memang
    # dibuat terpisah.
    # ========================================================

    train_manifest_path = Path(
        "manifests/train_replay_manifest.csv"
    )

    if train_manifest_path.exists():

        train_manifest = pd.read_csv(
            train_manifest_path
        )

        train_source_ids = set(
            train_manifest[
                "source_file_id"
            ]
            .astype(str)
        )

        unseen_source_ids = set(
            unseen[
                "source_file_id"
            ]
            .astype(str)
        )

        overlap = (
            train_source_ids
            &
            unseen_source_ids
        )

        print()
        print(
            "Source overlap dengan training:",
            len(overlap)
        )

        if overlap:

            raise RuntimeError(
                "LEAKAGE: source simulated unseen "
                "muncul di training."
            )

    # ========================================================
    # LOAD TRAINING STATS
    # ========================================================

    stats = torch.load(
        STATS_PATH,
        map_location="cpu",
        weights_only=False
    )

    print()
    print(
        "Training statistics loaded:"
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
    #
    # Threshold untuk F1/BAcc/accuracy tidak dicari pada test.
    # Threshold diambil dari validation seen masing-masing model.
    # ========================================================

    (
        no_grl_val_eer,
        no_grl_threshold
    ) = get_validation_eer_threshold(
        NO_GRL_VAL_SCORES
    )

    (
        with_grl_val_eer,
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
        f"EER={no_grl_val_eer * 100:.2f}%",
        f"threshold={no_grl_threshold:.8f}"
    )

    print(
        "WITH GRL:",
        f"EER={with_grl_val_eer * 100:.2f}%",
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
    # DATASET
    # ========================================================

    dataset = UnseenReplayDataset(

        dataframe=
            unseen,

        feature_extractor=
            feature_extractor,

        stats=
            stats
    )

    loader = DataLoader(

        dataset,

        batch_size=8,

        shuffle=False,

        num_workers=0,

        pin_memory=False
    )

    # ========================================================
    # LOAD MODELS
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
    # EVALUATE NO GRL
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
    # EVALUATE WITH GRL
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
    # CREATE OUTPUT DIR
    # ========================================================

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE PER-MODEL SCORES
    # ========================================================

    no_grl_score_path = (
        RESULT_DIR
        /
        "scores_unseen_no_grl.csv"
    )

    with_grl_score_path = (
        RESULT_DIR
        /
        "scores_unseen_with_grl.csv"
    )

    combined_score_path = (
        RESULT_DIR
        /
        "scores_simulated_unseen.csv"
    )

    pd.DataFrame(
        no_grl_rows
    ).to_csv(
        no_grl_score_path,
        index=False
    )

    pd.DataFrame(
        with_grl_rows
    ).to_csv(
        with_grl_score_path,
        index=False
    )

    combined_scores = pd.DataFrame(
        no_grl_rows
        +
        with_grl_rows
    )

    combined_scores.to_csv(
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

            "simulated_unseen_eer":
                no_grl_metrics[
                    "eer"
                ],

            "simulated_unseen_auc":
                no_grl_metrics[
                    "auc"
                ],

            "simulated_unseen_f1":
                no_grl_metrics[
                    "f1"
                ],

            "simulated_unseen_balanced_accuracy":
                no_grl_metrics[
                    "balanced_accuracy"
                ],

            "simulated_unseen_accuracy":
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

            "simulated_unseen_eer":
                with_grl_metrics[
                    "eer"
                ],

            "simulated_unseen_auc":
                with_grl_metrics[
                    "auc"
                ],

            "simulated_unseen_f1":
                with_grl_metrics[
                    "f1"
                ],

            "simulated_unseen_balanced_accuracy":
                with_grl_metrics[
                    "balanced_accuracy"
                ],

            "simulated_unseen_accuracy":
                with_grl_metrics[
                    "accuracy"
                ]
        }
    ])

    summary_path = (
        RESULT_DIR
        /
        "evaluation_simulated_unseen_summary.csv"
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

    condition_table = (
        build_condition_table(
            combined_scores,
            thresholds
        )
    )

    condition_path = (
        RESULT_DIR
        /
        "evaluation_simulated_unseen_conditions.csv"
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
            "simulated_unseen_rir",

        "manifest":
            str(
                SIM_MANIFEST
            ),

        "test_rows":
            len(
                unseen
            ),

        "label_0":
            int(
                (
                    unseen["label"]
                    ==
                    0
                ).sum()
            ),

        "label_1":
            int(
                (
                    unseen["label"]
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
            )
    }

    config_path = (
        RESULT_DIR
        /
        "evaluation_simulated_unseen_config.json"
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
        "SIMULATED UNSEEN RIR EVALUATION SELESAI"
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
        "Unseen EER   :",
        f"{no_grl_metrics['eer'] * 100:.2f}%"
    )

    print(
        "Unseen AUC   :",
        f"{no_grl_metrics['auc']:.4f}"
    )

    print(
        "Unseen F1    :",
        f"{no_grl_metrics['f1']:.4f}"
    )

    print(
        "Unseen BAcc  :",
        f"{no_grl_metrics['balanced_accuracy']:.4f}"
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
        "Unseen EER   :",
        f"{with_grl_metrics['eer'] * 100:.2f}%"
    )

    print(
        "Unseen AUC   :",
        f"{with_grl_metrics['auc']:.4f}"
    )

    print(
        "Unseen F1    :",
        f"{with_grl_metrics['f1']:.4f}"
    )

    print(
        "Unseen BAcc  :",
        f"{with_grl_metrics['balanced_accuracy']:.4f}"
    )

    # ========================================================
    # GRL DIFFERENCE
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
        "Perbedaan EER:"
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
            "WITH GRL menghasilkan EER lebih rendah "
            "pada simulated unseen RIR."
        )

        print(
            "Ini mendukung adanya peningkatan "
            "generalisasi ke RIR unseen."
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
            "WITH GRL menghasilkan EER lebih tinggi "
            "pada simulated unseen RIR."
        )

        print(
            "Pada kondisi ini GRL belum menunjukkan "
            "keuntungan generalisasi."
        )

    else:

        print(
            "HASIL SEMENTARA:"
        )

        print(
            "NO GRL dan WITH GRL menghasilkan "
            "EER simulated unseen yang sama."
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