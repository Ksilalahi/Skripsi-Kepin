from pathlib import Path
import argparse
import hashlib
import json
import random

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

DEFAULT_MANIFEST = Path(
    "manifests/train_replay_manifest.csv"
)

CHECKPOINT_DIR = Path(
    "checkpoints"
)

RESULT_DIR = Path(
    "results"
)

PROCESSED_DIR = Path(
    "data/processed"
)

N_DOMAIN = 5
EMBEDDING_DIM = 256

LAMBDA_TRANSFORM = 0.10


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# SHA256
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


# ============================================================
# AUDIO LOADER
# ============================================================

def load_audio_4s(
    path,
    training=False,
    target_sr=TARGET_SR,
    seconds=SECONDS
):

    path = Path(
        str(path)
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Audio tidak ditemukan: "
            f"{path}"
        )

    audio, sr = sf.read(
        str(path),
        always_2d=True
    )

    # ========================================================
    # MULTICHANNEL -> MONO
    # ========================================================

    audio = audio.mean(
        axis=1
    )

    # ========================================================
    # RESAMPLE
    # ========================================================

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
    # CROP
    # ========================================================

    elif n > target_length:

        if training:

            max_start = (
                n
                -
                target_length
            )

            start = np.random.randint(
                0,
                max_start + 1
            )

        else:

            start = (
                n
                -
                target_length
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
# METRICS
# ============================================================

def compute_metrics(
    y_true,
    fake_score,
    threshold=0.5
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    fake_score = np.asarray(
        fake_score
    ).astype(float)

    if len(
        np.unique(
            y_true
        )
    ) < 2:

        raise RuntimeError(
            "Validation membutuhkan "
            "label 0 dan 1."
        )

    # ========================================================
    # EER
    # ========================================================

    fpr, tpr, thresholds = roc_curve(
        y_true,
        fake_score,
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

    eer_threshold = float(
        thresholds[
            index
        ]
    )

    # ========================================================
    # FIXED 0.5 METRICS DURING VALIDATION
    # ========================================================

    prediction = (
        fake_score
        >=
        threshold
    ).astype(int)

    auc = float(
        roc_auc_score(
            y_true,
            fake_score
        )
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
# DATASET
# ============================================================

class ReplayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        feature_extractor,
        stats=None,
        training=False,
        feature_set="full"
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

        self.training = training

        self.feature_set = (
            feature_set
        )

        valid_feature_sets = [
            "logmel",
            "logmel_phase",
            "full"
        ]

        if (
            self.feature_set
            not in
            valid_feature_sets
        ):

            raise ValueError(
                "Feature set tidak valid: "
                f"{self.feature_set}"
            )

    def __len__(self):

        return len(
            self.df
        )

    def extract_features(
        self,
        wav
    ):

        # ====================================================
        # wav:
        # [1, T]
        #
        # SpectralPhase:
        # [B, 1, T]
        # ====================================================

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

        # ====================================================
        # [1,80,T] -> [80,T]
        # ====================================================

        logmel = (
            logmel.squeeze(0)
        )

        phase = (
            phase.squeeze(0)
        )

        mgd = (
            mgd.squeeze(0)
        )

        # ====================================================
        # NUMERICAL CHECK
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
                    f"{name} memiliki "
                    "NaN atau Inf."
                )

        # ====================================================
        # TRAINING-ONLY STANDARDIZATION
        # ====================================================

        if self.stats is not None:

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

        # ====================================================
        # FEATURE ABLATION
        #
        # Shape model tetap 3-channel supaya backbone
        # dan jumlah parameter tetap sama.
        #
        # logmel:
        # [logmel, 0, 0]
        #
        # logmel_phase:
        # [logmel, phase, 0]
        #
        # full:
        # [logmel, phase, mgd]
        # ====================================================

        zero_phase = torch.zeros_like(
            phase
        )

        zero_mgd = torch.zeros_like(
            mgd
        )

        if (
            self.feature_set
            ==
            "logmel"
        ):

            features = torch.stack(
                [
                    logmel,
                    zero_phase,
                    zero_mgd
                ],
                dim=0
            )

        elif (
            self.feature_set
            ==
            "logmel_phase"
        ):

            features = torch.stack(
                [
                    logmel,
                    phase,
                    zero_mgd
                ],
                dim=0
            )

        elif (
            self.feature_set
            ==
            "full"
        ):

            features = torch.stack(
                [
                    logmel,
                    phase,
                    mgd
                ],
                dim=0
            )

        else:

            raise RuntimeError(
                "Feature set tidak dikenal."
            )

        # Expected:
        # [3, 80, 397]

        return features.float()

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        wav = load_audio_4s(
            row[
                "file_path"
            ],
            training=
                self.training
        )

        features = (
            self.extract_features(
                wav
            )
        )

        return {

            "features":
                features,

            "label":
                torch.tensor(
                    int(
                        row[
                            "label"
                        ]
                    ),
                    dtype=torch.long
                ),

            "transform_label":
                torch.tensor(
                    int(
                        row[
                            "transform_label"
                        ]
                    ),
                    dtype=torch.long
                ),

            "domain_label":
                torch.tensor(
                    int(
                        row[
                            "domain_label"
                        ]
                    ),
                    dtype=torch.long
                ),

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

            "condition":
                str(
                    row[
                        "condition"
                    ]
                )
        }


# ============================================================
# COMPUTE TRAINING FEATURE STATISTICS
# ============================================================

def compute_training_stats(
    train_df,
    feature_extractor,
    manifest_hash,
    stats_path,
    force=False
):

    # ========================================================
    # REUSE EXISTING STATS
    # ========================================================

    if (
        stats_path.exists()
        and
        not force
    ):

        saved = torch.load(
            stats_path,
            map_location="cpu",
            weights_only=False
        )

        if (
            saved.get(
                "manifest_hash"
            )
            ==
            manifest_hash
        ):

            print()
            print(
                "Training feature stats ditemukan."
            )

            print(
                "Menggunakan:",
                stats_path
            )

            return saved

        print()
        print(
            "Stats berasal dari manifest berbeda."
        )

        print(
            "Stats akan dihitung ulang."
        )

    # ========================================================
    # COMPUTE NEW STATS
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "MENGHITUNG TRAINING FEATURE STATISTICS"
    )

    print(
        "=============================================="
    )

    logmel_sum = 0.0
    logmel_sq_sum = 0.0
    logmel_count = 0

    phase_sum = 0.0
    phase_sq_sum = 0.0
    phase_count = 0

    mgd_sum = 0.0
    mgd_sq_sum = 0.0
    mgd_count = 0

    total = len(
        train_df
    )

    train_reset = (
        train_df
        .reset_index(
            drop=True
        )
    )

    for i, row in train_reset.iterrows():

        wav = load_audio_4s(
            row[
                "file_path"
            ],
            training=False
        )

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

        logmel = (
            logmel.double()
        )

        phase = (
            phase.double()
        )

        mgd = (
            mgd.double()
        )

        # ====================================================
        # FINITE CHECK
        # ====================================================

        if not torch.isfinite(
            logmel
        ).all():

            raise RuntimeError(
                "Log-Mel NaN/Inf: "
                f"{row['file_id']}"
            )

        if not torch.isfinite(
            phase
        ).all():

            raise RuntimeError(
                "Phase NaN/Inf: "
                f"{row['file_id']}"
            )

        if not torch.isfinite(
            mgd
        ).all():

            raise RuntimeError(
                "MGD NaN/Inf: "
                f"{row['file_id']}"
            )

        # ====================================================
        # ACCUMULATE LOGMEL
        # ====================================================

        logmel_sum += float(
            logmel.sum()
        )

        logmel_sq_sum += float(
            (
                logmel ** 2
            ).sum()
        )

        logmel_count += (
            logmel.numel()
        )

        # ====================================================
        # ACCUMULATE PHASE
        # ====================================================

        phase_sum += float(
            phase.sum()
        )

        phase_sq_sum += float(
            (
                phase ** 2
            ).sum()
        )

        phase_count += (
            phase.numel()
        )

        # ====================================================
        # ACCUMULATE MGD
        # ====================================================

        mgd_sum += float(
            mgd.sum()
        )

        mgd_sq_sum += float(
            (
                mgd ** 2
            ).sum()
        )

        mgd_count += (
            mgd.numel()
        )

        if (
            (i + 1) % 50 == 0
            or
            (i + 1) == total
        ):

            print(
                f"[{i + 1}/{total}] "
                "training samples"
            )

    # ========================================================
    # LOGMEL MEAN / STD
    # ========================================================

    logmel_mean = (
        logmel_sum
        /
        logmel_count
    )

    logmel_var = (
        logmel_sq_sum
        /
        logmel_count
        -
        logmel_mean ** 2
    )

    logmel_std = float(
        np.sqrt(
            max(
                logmel_var,
                1e-12
            )
        )
    )

    # ========================================================
    # PHASE MEAN / STD
    # ========================================================

    phase_mean = (
        phase_sum
        /
        phase_count
    )

    phase_var = (
        phase_sq_sum
        /
        phase_count
        -
        phase_mean ** 2
    )

    phase_std = float(
        np.sqrt(
            max(
                phase_var,
                1e-12
            )
        )
    )

    # ========================================================
    # MGD MEAN / STD
    # ========================================================

    mgd_mean = (
        mgd_sum
        /
        mgd_count
    )

    mgd_var = (
        mgd_sq_sum
        /
        mgd_count
        -
        mgd_mean ** 2
    )

    mgd_std = float(
        np.sqrt(
            max(
                mgd_var,
                1e-12
            )
        )
    )

    stats = {

        "manifest_hash":
            manifest_hash,

        "train_rows":
            len(
                train_df
            ),

        "logmel_mean":
            float(
                logmel_mean
            ),

        "logmel_std":
            float(
                logmel_std
            ),

        "phase_mean":
            float(
                phase_mean
            ),

        "phase_std":
            float(
                phase_std
            ),

        "mgd_mean":
            float(
                mgd_mean
            ),

        "mgd_std":
            float(
                mgd_std
            )
    }

    stats_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        stats,
        stats_path
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

    print()
    print(
        "Saved:",
        stats_path
    )

    return stats


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    use_grl=False,
    lambda_domain=0.05,
    grl_strength=1.0
):

    model.train()

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_transform_loss = 0.0
    running_domain_loss = 0.0

    correct_class = 0
    correct_transform = 0

    domain_correct = 0
    domain_total = 0

    total = 0

    for batch in loader:

        x = batch[
            "features"
        ].to(device)

        y = batch[
            "label"
        ].to(device)

        transform_y = batch[
            "transform_label"
        ].to(device)

        domain_y = batch[
            "domain_label"
        ].to(device)

        optimizer.zero_grad()

        # ====================================================
        # GRL STRENGTH
        # ====================================================

        effective_grl = (
            grl_strength
            if use_grl
            else 0.0
        )

        # ====================================================
        # FORWARD
        # ====================================================

        (
            class_logits,
            transform_logits,
            domain_logits,
            _
        ) = model(
            x,
            grl_strength=
                effective_grl
        )

        # ====================================================
        # CLASS LOSS
        # ====================================================

        loss_cls = (
            F.cross_entropy(
                class_logits,
                y
            )
        )

        # ====================================================
        # TRANSFORM LOSS
        # ====================================================

        loss_transform = (
            F.cross_entropy(
                transform_logits,
                transform_y
            )
        )

        # ====================================================
        # MASKED DOMAIN LOSS
        # ====================================================

        valid_domain = (
            domain_y >= 0
        )

        if (
            use_grl
            and
            valid_domain.any()
        ):

            loss_domain = (
                F.cross_entropy(
                    domain_logits[
                        valid_domain
                    ],
                    domain_y[
                        valid_domain
                    ]
                )
            )

        else:

            loss_domain = (
                class_logits.sum()
                *
                0.0
            )

        # ====================================================
        # TOTAL LOSS
        # ====================================================

        loss = (
            loss_cls
            +
            LAMBDA_TRANSFORM
            *
            loss_transform
        )

        if use_grl:

            loss = (
                loss
                +
                lambda_domain
                *
                loss_domain
            )

        # ====================================================
        # BACKWARD
        # ====================================================

        loss.backward()

        optimizer.step()

        batch_size = (
            y.size(0)
        )

        total += (
            batch_size
        )

        # ====================================================
        # LOSSES
        # ====================================================

        running_total_loss += (
            float(
                loss.item()
            )
            *
            batch_size
        )

        running_cls_loss += (
            float(
                loss_cls.item()
            )
            *
            batch_size
        )

        running_transform_loss += (
            float(
                loss_transform.item()
            )
            *
            batch_size
        )

        # ====================================================
        # ACCURACY
        # ====================================================

        class_pred = (
            class_logits.argmax(
                dim=1
            )
        )

        transform_pred = (
            transform_logits.argmax(
                dim=1
            )
        )

        correct_class += int(
            (
                class_pred
                ==
                y
            )
            .sum()
            .item()
        )

        correct_transform += int(
            (
                transform_pred
                ==
                transform_y
            )
            .sum()
            .item()
        )

        # ====================================================
        # DOMAIN METRICS
        # ====================================================

        if (
            use_grl
            and
            valid_domain.any()
        ):

            valid_count = int(
                valid_domain
                .sum()
                .item()
            )

            running_domain_loss += (
                float(
                    loss_domain.item()
                )
                *
                valid_count
            )

            domain_pred = (
                domain_logits[
                    valid_domain
                ]
                .argmax(
                    dim=1
                )
            )

            domain_correct += int(
                (
                    domain_pred
                    ==
                    domain_y[
                        valid_domain
                    ]
                )
                .sum()
                .item()
            )

            domain_total += (
                valid_count
            )

    return {

        "loss":
            running_total_loss
            /
            total,

        "class_loss":
            running_cls_loss
            /
            total,

        "transform_loss":
            running_transform_loss
            /
            total,

        "domain_loss":
            (
                running_domain_loss
                /
                domain_total
                if domain_total > 0
                else 0.0
            ),

        "class_accuracy":
            correct_class
            /
            total,

        "transform_accuracy":
            correct_transform
            /
            total,

        "domain_accuracy":
            (
                domain_correct
                /
                domain_total
                if domain_total > 0
                else 0.0
            ),

        "domain_samples":
            domain_total
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    use_grl=False,
    lambda_domain=0.05,
    grl_strength=1.0
):

    model.eval()

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_transform_loss = 0.0
    running_domain_loss = 0.0

    correct_class = 0
    correct_transform = 0

    domain_correct = 0
    domain_total = 0

    total = 0

    y_true = []
    fake_scores = []

    score_rows = []

    for batch in loader:

        x = batch[
            "features"
        ].to(device)

        y = batch[
            "label"
        ].to(device)

        transform_y = batch[
            "transform_label"
        ].to(device)

        domain_y = batch[
            "domain_label"
        ].to(device)

        effective_grl = (
            grl_strength
            if use_grl
            else 0.0
        )

        (
            class_logits,
            transform_logits,
            domain_logits,
            _
        ) = model(
            x,
            grl_strength=
                effective_grl
        )

        # ====================================================
        # LOSSES
        # ====================================================

        loss_cls = (
            F.cross_entropy(
                class_logits,
                y
            )
        )

        loss_transform = (
            F.cross_entropy(
                transform_logits,
                transform_y
            )
        )

        valid_domain = (
            domain_y >= 0
        )

        if (
            use_grl
            and
            valid_domain.any()
        ):

            loss_domain = (
                F.cross_entropy(
                    domain_logits[
                        valid_domain
                    ],
                    domain_y[
                        valid_domain
                    ]
                )
            )

        else:

            loss_domain = (
                class_logits.sum()
                *
                0.0
            )

        loss = (
            loss_cls
            +
            LAMBDA_TRANSFORM
            *
            loss_transform
        )

        if use_grl:

            loss = (
                loss
                +
                lambda_domain
                *
                loss_domain
            )

        batch_size = (
            y.size(0)
        )

        total += (
            batch_size
        )

        running_total_loss += (
            float(
                loss.item()
            )
            *
            batch_size
        )

        running_cls_loss += (
            float(
                loss_cls.item()
            )
            *
            batch_size
        )

        running_transform_loss += (
            float(
                loss_transform.item()
            )
            *
            batch_size
        )

        # ====================================================
        # CLASS / TRANSFORM ACC
        # ====================================================

        class_pred = (
            class_logits.argmax(
                dim=1
            )
        )

        transform_pred = (
            transform_logits.argmax(
                dim=1
            )
        )

        correct_class += int(
            (
                class_pred
                ==
                y
            )
            .sum()
            .item()
        )

        correct_transform += int(
            (
                transform_pred
                ==
                transform_y
            )
            .sum()
            .item()
        )

        # ====================================================
        # DOMAIN METRICS
        # ====================================================

        if (
            use_grl
            and
            valid_domain.any()
        ):

            valid_count = int(
                valid_domain
                .sum()
                .item()
            )

            running_domain_loss += (
                float(
                    loss_domain.item()
                )
                *
                valid_count
            )

            domain_pred = (
                domain_logits[
                    valid_domain
                ]
                .argmax(
                    dim=1
                )
            )

            domain_correct += int(
                (
                    domain_pred
                    ==
                    domain_y[
                        valid_domain
                    ]
                )
                .sum()
                .item()
            )

            domain_total += (
                valid_count
            )

        # ====================================================
        # CONTINUOUS FAKE SCORE
        # ====================================================

        probability = torch.softmax(
            class_logits,
            dim=1
        )

        fake_probability = (
            probability[
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

        y_true.extend(
            y_numpy.tolist()
        )

        fake_scores.extend(
            fake_probability.tolist()
        )

        for i in range(
            batch_size
        ):

            score_rows.append({

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
                        y_numpy[
                            i
                        ]
                    ),

                "fake_score":
                    float(
                        fake_probability[
                            i
                        ]
                    ),

                "condition":
                    batch[
                        "condition"
                    ][i]
            })

    classification_metrics = (
        compute_metrics(
            y_true,
            fake_scores
        )
    )

    return {

        "loss":
            running_total_loss
            /
            total,

        "class_loss":
            running_cls_loss
            /
            total,

        "transform_loss":
            running_transform_loss
            /
            total,

        "domain_loss":
            (
                running_domain_loss
                /
                domain_total
                if domain_total > 0
                else 0.0
            ),

        "class_accuracy":
            correct_class
            /
            total,

        "transform_accuracy":
            correct_transform
            /
            total,

        "domain_accuracy":
            (
                domain_correct
                /
                domain_total
                if domain_total > 0
                else 0.0
            ),

        "domain_samples":
            domain_total,

        "eer":
            classification_metrics[
                "eer"
            ],

        "eer_threshold":
            classification_metrics[
                "eer_threshold"
            ],

        "auc":
            classification_metrics[
                "auc"
            ],

        "f1":
            classification_metrics[
                "f1"
            ],

        "balanced_accuracy":
            classification_metrics[
                "balanced_accuracy"
            ],

        "accuracy":
            classification_metrics[
                "accuracy"
            ],

        "score_rows":
            score_rows
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        default=str(
            DEFAULT_MANIFEST
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    parser.add_argument(
        "--recompute-stats",
        action="store_true"
    )

    # ========================================================
    # GRL OPTIONS
    # ========================================================

    parser.add_argument(
        "--use-grl",
        action="store_true"
    )

    parser.add_argument(
        "--lambda-domain",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--grl-strength",
        type=float,
        default=1.0
    )

    # ========================================================
    # EXPERIMENT NAME
    # ========================================================

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None
    )

    # ========================================================
    # FEATURE ABLATION
    # ========================================================

    parser.add_argument(
        "--feature-set",
        type=str,
        choices=[
            "logmel",
            "logmel_phase",
            "full"
        ],
        default="full"
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE ARGUMENTS
    # ========================================================

    if args.epochs <= 0:

        raise ValueError(
            "epochs harus > 0."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "batch-size harus > 0."
        )

    if args.lr <= 0:

        raise ValueError(
            "learning rate harus > 0."
        )

    if args.lambda_domain < 0:

        raise ValueError(
            "lambda-domain "
            "tidak boleh negatif."
        )

    if args.grl_strength < 0:

        raise ValueError(
            "grl-strength "
            "tidak boleh negatif."
        )

    # ========================================================
    # SEED
    # ========================================================

    set_seed(
        args.seed
    )

    # ========================================================
    # EXPERIMENT NAME
    # ========================================================

    if (
        args.experiment_name
        is not None
    ):

        experiment_name = (
            args.experiment_name
            .strip()
        )

        if experiment_name == "":

            raise ValueError(
                "experiment-name "
                "tidak boleh kosong."
            )

    else:

        # ----------------------------------------------------
        # FULL FEATURE DEFAULT EXPERIMENT
        # ----------------------------------------------------

        if (
            args.feature_set
            ==
            "full"
        ):

            if args.use_grl:

                experiment_name = (
                    "with_grl"
                )

            else:

                experiment_name = (
                    "no_grl"
                )

        # ----------------------------------------------------
        # ABLATION
        # ----------------------------------------------------

        else:

            if args.use_grl:

                experiment_name = (
                    f"ablation_"
                    f"{args.feature_set}_"
                    f"with_grl"
                )

            else:

                experiment_name = (
                    f"ablation_"
                    f"{args.feature_set}_"
                    f"no_grl"
                )

    # ========================================================
    # EXPERIMENT-SPECIFIC STATS
    # ========================================================

    stats_path = (
        PROCESSED_DIR
        /
        (
            "three_channel_stats_"
            f"{experiment_name}.pt"
        )
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "TRAIN REPLAY MODEL"
    )

    print(
        "=============================================="
    )

    print(
        "Experiment:",
        experiment_name
    )

    print(
        "Device:",
        device
    )

    print(
        "Seed:",
        args.seed
    )

    print(
        "Epochs:",
        args.epochs
    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "Learning rate:",
        args.lr
    )

    print(
        "Feature set:",
        args.feature_set
    )

    print(
        "GRL:",
        (
            "ON"
            if args.use_grl
            else "OFF"
        )
    )

    print(
        "GRL strength:",
        (
            args.grl_strength
            if args.use_grl
            else 0.0
        )
    )

    print(
        "Lambda domain:",
        (
            args.lambda_domain
            if args.use_grl
            else 0.0
        )
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest_path = Path(
        args.manifest
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            manifest_path
        )

    df = pd.read_csv(
        manifest_path
    )

    required_columns = [

        "file_id",
        "source_file_id",
        "file_path",
        "label",
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
            "Manifest kehilangan kolom: "
            f"{missing}"
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    df[
        "split"
    ] = (
        df[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
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
        "transform_label"
    ] = pd.to_numeric(
        df[
            "transform_label"
        ],
        errors="raise"
    ).astype(int)

    df[
        "domain_label"
    ] = pd.to_numeric(
        df[
            "domain_label"
        ],
        errors="raise"
    ).astype(int)

    # ========================================================
    # SPLIT
    # ========================================================

    train_df = df[
        df[
            "split"
        ]
        ==
        "train"
    ].copy()

    validation_df = df[
        df[
            "split"
        ]
        ==
        "validation"
    ].copy()

    print()
    print(
        "Manifest:",
        manifest_path
    )

    print(
        "Manifest rows:",
        len(df)
    )

    print(
        "Train rows:",
        len(train_df)
    )

    print(
        "Validation rows:",
        len(validation_df)
    )

    # ========================================================
    # FLEXIBLE SIZE VALIDATION
    # ========================================================

    if len(
        train_df
    ) == 0:

        raise RuntimeError(
            "Train split kosong."
        )

    if len(
        validation_df
    ) == 0:

        raise RuntimeError(
            "Validation split kosong."
        )

    print(
        "Train rows valid:",
        len(train_df)
    )

    print(
        "Validation rows valid:",
        len(validation_df)
    )

    # ========================================================
    # LABEL CHECK
    # ========================================================

    if not df[
        "label"
    ].isin(
        [
            0,
            1
        ]
    ).all():

        raise RuntimeError(
            "Label harus 0 atau 1."
        )

    if not df[
        "transform_label"
    ].isin(
        [
            0,
            1
        ]
    ).all():

        raise RuntimeError(
            "Transform label harus "
            "0 atau 1."
        )

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    print()
    print(
        "Train label distribution:"
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
        "Validation label distribution:"
    )

    print(
        validation_df[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    # Both labels must exist.
    if (
        train_df[
            "label"
        ].nunique()
        !=
        2
    ):

        raise RuntimeError(
            "Train harus memiliki "
            "label 0 dan 1."
        )

    if (
        validation_df[
            "label"
        ].nunique()
        !=
        2
    ):

        raise RuntimeError(
            "Validation harus memiliki "
            "label 0 dan 1."
        )

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
        len(
            overlap
        )
    )

    if overlap:

        print(
            "Contoh overlap:"
        )

        for source_id in list(
            overlap
        )[:10]:

            print(
                source_id
            )

        raise RuntimeError(
            "Source leakage ditemukan."
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    duplicate_file_id = int(
        df[
            "file_id"
        ]
        .duplicated()
        .sum()
    )

    duplicate_file_path = int(
        df[
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
        duplicate_file_path
    )

    if duplicate_file_id > 0:

        raise RuntimeError(
            "Duplicate file_id ditemukan."
        )

    if duplicate_file_path > 0:

        raise RuntimeError(
            "Duplicate file_path ditemukan."
        )

    # ========================================================
    # DOMAIN AUDIT
    # ========================================================

    valid_train_domain = train_df[
        train_df[
            "domain_label"
        ]
        >=
        0
    ]

    valid_validation_domain = validation_df[
        validation_df[
            "domain_label"
        ]
        >=
        0
    ]

    print()
    print(
        "Train domain samples:",
        len(
            valid_train_domain
        )
    )

    print(
        "Validation domain samples:",
        len(
            valid_validation_domain
        )
    )

    if args.use_grl:

        if len(
            valid_train_domain
        ) == 0:

            raise RuntimeError(
                "GRL aktif tetapi "
                "tidak ada domain label "
                "valid pada training."
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
    # MANIFEST HASH
    # ========================================================

    manifest_hash = sha256_file(
        manifest_path
    )

    # ========================================================
    # TRAINING-ONLY STATISTICS
    # ========================================================

    stats = compute_training_stats(

        train_df=
            train_df,

        feature_extractor=
            feature_extractor,

        manifest_hash=
            manifest_hash,

        stats_path=
            stats_path,

        force=
            args.recompute_stats
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = ReplayDataset(

        dataframe=
            train_df,

        feature_extractor=
            feature_extractor,

        stats=
            stats,

        training=True,

        feature_set=
            args.feature_set
    )

    validation_dataset = ReplayDataset(

        dataframe=
            validation_df,

        feature_extractor=
            feature_extractor,

        stats=
            stats,

        training=False,

        feature_set=
            args.feature_set
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=
            args.batch_size,

        shuffle=True,

        num_workers=0,

        pin_memory=False
    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=
            args.batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=False
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = ReplayResNet(

        n_transform=2,

        n_domain=
            N_DOMAIN,

        embedding_dim=
            EMBEDDING_DIM
    )

    model = model.to(
        device
    )

    print()
    print(
        "Model:",
        model.__class__.__name__
    )

    # ========================================================
    # LOSS INFO
    # ========================================================

    if args.use_grl:

        print(
            "GRL/domain loss: ON"
        )

        print(
            "Loss:"
        )

        print(
            "L = L_cls "
            f"+ {LAMBDA_TRANSFORM:.2f} "
            "* L_transform "
            f"+ {args.lambda_domain:.2f} "
            "* L_domain"
        )

    else:

        print(
            "GRL/domain loss: OFF"
        )

        print(
            "Loss:"
        )

        print(
            "L = L_cls "
            f"+ {LAMBDA_TRANSFORM:.2f} "
            "* L_transform"
        )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=
            args.lr,

        weight_decay=
            args.weight_decay
    )

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    checkpoint_path = (
        CHECKPOINT_DIR
        /
        f"replay_{experiment_name}.pt"
    )

    history_path = (
        RESULT_DIR
        /
        f"history_{experiment_name}.csv"
    )

    scores_path = (
        RESULT_DIR
        /
        (
            f"scores_{experiment_name}"
            "_validation.csv"
        )
    )

    config_path = (
        RESULT_DIR
        /
        f"config_{experiment_name}.json"
    )

    # ========================================================
    # FEATURE CHANNEL DESCRIPTION
    # ========================================================

    if (
        args.feature_set
        ==
        "logmel"
    ):

        feature_channels = [
            "logmel"
        ]

    elif (
        args.feature_set
        ==
        "logmel_phase"
    ):

        feature_channels = [
            "logmel",
            "phase"
        ]

    else:

        feature_channels = [
            "logmel",
            "phase",
            "mgd"
        ]

    # ========================================================
    # CONFIG
    # ========================================================

    config = {

        "experiment":
            experiment_name,

        "manifest":
            str(
                manifest_path
            ),

        "manifest_sha256":
            manifest_hash,

        "stats_path":
            str(
                stats_path
            ),

        "seed":
            args.seed,

        "epochs":
            args.epochs,

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.lr,

        "weight_decay":
            args.weight_decay,

        "sample_rate":
            TARGET_SR,

        "audio_seconds":
            SECONDS,

        "n_fft":
            512,

        "hop":
            160,

        "n_mels":
            80,

        "feature_set":
            args.feature_set,

        "feature_channels":
            feature_channels,

        "physical_input_channels":
            3,

        "embedding_dim":
            EMBEDDING_DIM,

        "lambda_transform":
            LAMBDA_TRANSFORM,

        "lambda_domain":
            (
                args.lambda_domain
                if args.use_grl
                else 0.0
            ),

        "use_grl":
            args.use_grl,

        "grl_strength":
            (
                args.grl_strength
                if args.use_grl
                else 0.0
            ),

        "train_rows":
            len(
                train_df
            ),

        "validation_rows":
            len(
                validation_df
            ),

        "train_domain_samples":
            len(
                valid_train_domain
            ),

        "validation_domain_samples":
            len(
                valid_validation_domain
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
    # TRAINING LOOP
    # ========================================================

    history = []

    best_eer = float(
        "inf"
    )

    best_epoch = -1

    print()
    print(
        "=============================================="
    )

    print(
        "TRAINING START"
    )

    print(
        "=============================================="
    )

    for epoch in range(
        1,
        args.epochs + 1
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        train_result = (
            train_one_epoch(

                model=
                    model,

                loader=
                    train_loader,

                optimizer=
                    optimizer,

                device=
                    device,

                use_grl=
                    args.use_grl,

                lambda_domain=
                    args.lambda_domain,

                grl_strength=
                    args.grl_strength
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        validation_result = (
            validate(

                model=
                    model,

                loader=
                    validation_loader,

                device=
                    device,

                use_grl=
                    args.use_grl,

                lambda_domain=
                    args.lambda_domain,

                grl_strength=
                    args.grl_strength
            )
        )

        # ====================================================
        # HISTORY ROW
        # ====================================================

        history_row = {

            "epoch":
                epoch,

            "train_loss":
                train_result[
                    "loss"
                ],

            "train_class_loss":
                train_result[
                    "class_loss"
                ],

            "train_transform_loss":
                train_result[
                    "transform_loss"
                ],

            "train_domain_loss":
                train_result[
                    "domain_loss"
                ],

            "train_class_accuracy":
                train_result[
                    "class_accuracy"
                ],

            "train_transform_accuracy":
                train_result[
                    "transform_accuracy"
                ],

            "train_domain_accuracy":
                train_result[
                    "domain_accuracy"
                ],

            "train_domain_samples":
                train_result[
                    "domain_samples"
                ],

            "validation_loss":
                validation_result[
                    "loss"
                ],

            "validation_class_loss":
                validation_result[
                    "class_loss"
                ],

            "validation_transform_loss":
                validation_result[
                    "transform_loss"
                ],

            "validation_domain_loss":
                validation_result[
                    "domain_loss"
                ],

            "validation_class_accuracy":
                validation_result[
                    "class_accuracy"
                ],

            "validation_transform_accuracy":
                validation_result[
                    "transform_accuracy"
                ],

            "validation_domain_accuracy":
                validation_result[
                    "domain_accuracy"
                ],

            "validation_domain_samples":
                validation_result[
                    "domain_samples"
                ],

            "validation_eer":
                validation_result[
                    "eer"
                ],

            "validation_eer_threshold":
                validation_result[
                    "eer_threshold"
                ],

            "validation_auc":
                validation_result[
                    "auc"
                ],

            "validation_f1":
                validation_result[
                    "f1"
                ],

            "validation_balanced_accuracy":
                validation_result[
                    "balanced_accuracy"
                ],

            "validation_accuracy":
                validation_result[
                    "accuracy"
                ]
        }

        history.append(
            history_row
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False
        )

        # ====================================================
        # PRINT EPOCH
        # ====================================================

        print()
        print(
            f"Epoch "
            f"{epoch:02d}/"
            f"{args.epochs:02d}"
        )

        print(
            "Train loss       :",
            f"{train_result['loss']:.6f}"
        )

        print(
            "Train class acc  :",
            f"{train_result['class_accuracy']:.4f}"
        )

        print(
            "Train trans acc  :",
            f"{train_result['transform_accuracy']:.4f}"
        )

        if args.use_grl:

            print(
                "Train domain loss:",
                f"{train_result['domain_loss']:.6f}"
            )

            print(
                "Train domain acc :",
                f"{train_result['domain_accuracy']:.4f}"
            )

        print(
            "Val loss         :",
            f"{validation_result['loss']:.6f}"
        )

        print(
            "Val class acc    :",
            f"{validation_result['class_accuracy']:.4f}"
        )

        print(
            "Val trans acc    :",
            f"{validation_result['transform_accuracy']:.4f}"
        )

        if args.use_grl:

            print(
                "Val domain loss :",
                f"{validation_result['domain_loss']:.6f}"
            )

            print(
                "Val domain acc  :",
                f"{validation_result['domain_accuracy']:.4f}"
            )

        print(
            "Val EER          :",
            f"{validation_result['eer'] * 100:.2f}%"
        )

        print(
            "Val EER threshold:",
            f"{validation_result['eer_threshold']:.8f}"
        )

        print(
            "Val AUC          :",
            f"{validation_result['auc']:.4f}"
        )

        print(
            "Val F1           :",
            f"{validation_result['f1']:.4f}"
        )

        print(
            "Val BAcc         :",
            f"{validation_result['balanced_accuracy']:.4f}"
        )

        # ====================================================
        # BEST CHECKPOINT BY VALIDATION EER
        # ====================================================

        if (
            validation_result[
                "eer"
            ]
            <
            best_eer
        ):

            best_eer = (
                validation_result[
                    "eer"
                ]
            )

            best_epoch = (
                epoch
            )

            checkpoint = {

                "experiment":
                    experiment_name,

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "validation_eer":
                    best_eer,

                "validation_eer_threshold":
                    validation_result[
                        "eer_threshold"
                    ],

                "validation_auc":
                    validation_result[
                        "auc"
                    ],

                "validation_f1":
                    validation_result[
                        "f1"
                    ],

                "validation_balanced_accuracy":
                    validation_result[
                        "balanced_accuracy"
                    ],

                "validation_accuracy":
                    validation_result[
                        "accuracy"
                    ],

                "validation_domain_accuracy":
                    validation_result[
                        "domain_accuracy"
                    ],

                "manifest_hash":
                    manifest_hash,

                "stats_path":
                    str(
                        stats_path
                    ),

                "stats":
                    stats,

                "feature_set":
                    args.feature_set,

                "config":
                    config
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            # =================================================
            # CONTINUOUS VALIDATION SCORES
            # =================================================

            score_df = pd.DataFrame(
                validation_result[
                    "score_rows"
                ]
            )

            score_df[
                "seed"
            ] = (
                args.seed
            )

            score_df[
                "epoch"
            ] = (
                epoch
            )

            score_df[
                "model"
            ] = (
                experiment_name
            )

            score_df[
                "feature_set"
            ] = (
                args.feature_set
            )

            score_df[
                "validation_eer_threshold"
            ] = (
                validation_result[
                    "eer_threshold"
                ]
            )

            score_df.to_csv(
                scores_path,
                index=False
            )

            print(
                "BEST CHECKPOINT SAVED"
            )

            print(
                " ->",
                checkpoint_path
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "TRAINING SELESAI"
    )

    print(
        "=============================================="
    )

    print(
        "Experiment:",
        experiment_name
    )

    print(
        "Feature set:",
        args.feature_set
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation EER:",
        f"{best_eer * 100:.2f}%"
    )

    print(
        "Checkpoint:",
        checkpoint_path
    )

    print(
        "History:",
        history_path
    )

    print(
        "Validation scores:",
        scores_path
    )

    print(
        "Config:",
        config_path
    )

    print(
        "Stats:",
        stats_path
    )

    print()

    if args.use_grl:

        print(
            "Loss:"
        )

        print(
            "L = L_cls "
            f"+ {LAMBDA_TRANSFORM:.2f} "
            "* L_transform "
            f"+ {args.lambda_domain:.2f} "
            "* L_domain"
        )

        print(
            "GRL = ON"
        )

    else:

        print(
            "Loss:"
        )

        print(
            "L = L_cls "
            f"+ {LAMBDA_TRANSFORM:.2f} "
            "* L_transform"
        )

        print(
            "GRL = OFF"
        )

if __name__ == "__main__":

    main()