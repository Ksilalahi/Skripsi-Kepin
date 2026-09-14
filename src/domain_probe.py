from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import resample_poly
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

MANIFEST_PATH = Path(
    "manifests/train_replay_manifest.csv"
)

NO_GRL_CHECKPOINT = Path(
    "checkpoints/replay_no_grl.pt"
)

WITH_GRL_CHECKPOINT = Path(
    "checkpoints/replay_with_grl.pt"
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
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# DATASET
# ============================================================

class DomainProbeDataset(Dataset):

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

        # [1,T]
        # menjadi [B,1,T]
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

        # [1,80,T]
        # -> [80,T]

        logmel = (
            logmel.squeeze(0)
        )

        phase = (
            phase.squeeze(0)
        )

        mgd = (
            mgd.squeeze(0)
        )

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
                    f"{name} memiliki "
                    f"NaN/Inf pada "
                    f"{row['file_id']}"
                )

        # ----------------------------------------------------
        # STANDARDIZATION
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

        # ----------------------------------------------------
        # THREE CHANNEL
        # ----------------------------------------------------

        features = torch.stack(
            [
                logmel,
                phase,
                mgd
            ],
            dim=0
        )

        return {

            "features":
                features.float(),

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
                )
        }


# ============================================================
# LINEAR DOMAIN PROBE
# ============================================================

class LinearDomainProbe(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        n_domain=5
    ):

        super().__init__()

        self.classifier = nn.Linear(
            embedding_dim,
            n_domain
        )

    def forward(
        self,
        x
    ):

        return self.classifier(
            x
        )


# ============================================================
# LOAD REPLAY MODEL
# ============================================================

def load_replay_model(
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

    # --------------------------------------------------------
    # FREEZE BACKBONE
    # --------------------------------------------------------

    for parameter in model.parameters():

        parameter.requires_grad = False

    return (
        model,
        checkpoint
    )


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device,
    label
):

    model.eval()

    embeddings = []
    domain_labels = []

    file_ids = []
    source_ids = []

    total = len(
        loader.dataset
    )

    processed = 0

    print()
    print(
        f"Extracting embeddings: {label}"
    )

    for batch in loader:

        x = batch[
            "features"
        ].to(device)

        # ----------------------------------------------------
        # IMPORTANT
        #
        # grl_strength=0 tidak mengubah embedding z.
        # Domain head tidak digunakan.
        # ----------------------------------------------------

        (
            _,
            _,
            _,
            z
        ) = model(
            x,
            grl_strength=0.0
        )

        embeddings.append(
            z.cpu()
        )

        domain_labels.append(
            batch[
                "domain_label"
            ].cpu()
        )

        file_ids.extend(
            batch[
                "file_id"
            ]
        )

        source_ids.extend(
            batch[
                "source_file_id"
            ]
        )

        processed += (
            x.size(0)
        )

        if (
            processed % 50 == 0
            or
            processed == total
        ):

            print(
                f"[{processed}/{total}]"
            )

    embeddings = torch.cat(
        embeddings,
        dim=0
    )

    domain_labels = torch.cat(
        domain_labels,
        dim=0
    )

    return {

        "embeddings":
            embeddings,

        "domain_labels":
            domain_labels,

        "file_ids":
            file_ids,

        "source_ids":
            source_ids
    }


# ============================================================
# TRAIN PROBE
# ============================================================

def train_domain_probe(
    train_embeddings,
    train_labels,
    validation_embeddings,
    validation_labels,
    device,
    seed,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4
):

    set_seed(
        seed
    )

    probe = LinearDomainProbe(
        embedding_dim=
            EMBEDDING_DIM,
        n_domain=
            N_DOMAIN
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=lr,
        weight_decay=
            weight_decay
    )

    train_x = (
        train_embeddings
        .to(device)
    )

    train_y = (
        train_labels
        .to(device)
    )

    validation_x = (
        validation_embeddings
        .to(device)
    )

    validation_y = (
        validation_labels
        .to(device)
    )

    best_validation_accuracy = -1.0
    best_validation_loss = float(
        "inf"
    )

    best_epoch = -1

    best_state = None

    history = []

    # ========================================================
    # TRAIN PROBE
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        probe.train()

        optimizer.zero_grad()

        train_logits = probe(
            train_x
        )

        train_loss = (
            F.cross_entropy(
                train_logits,
                train_y
            )
        )

        train_loss.backward()

        optimizer.step()

        train_prediction = (
            train_logits.argmax(
                dim=1
            )
        )

        train_accuracy = float(
            (
                train_prediction
                ==
                train_y
            )
            .float()
            .mean()
            .item()
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        probe.eval()

        with torch.no_grad():

            validation_logits = (
                probe(
                    validation_x
                )
            )

            validation_loss = (
                F.cross_entropy(
                    validation_logits,
                    validation_y
                )
            )

            validation_prediction = (
                validation_logits.argmax(
                    dim=1
                )
            )

            validation_accuracy = float(
                (
                    validation_prediction
                    ==
                    validation_y
                )
                .float()
                .mean()
                .item()
            )

        history.append({

            "epoch":
                epoch,

            "train_loss":
                float(
                    train_loss.item()
                ),

            "train_accuracy":
                train_accuracy,

            "validation_loss":
                float(
                    validation_loss.item()
                ),

            "validation_accuracy":
                validation_accuracy
        })

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        better = False

        if (
            validation_accuracy
            >
            best_validation_accuracy
        ):

            better = True

        elif (
            validation_accuracy
            ==
            best_validation_accuracy
            and
            float(
                validation_loss.item()
            )
            <
            best_validation_loss
        ):

            better = True

        if better:

            best_validation_accuracy = (
                validation_accuracy
            )

            best_validation_loss = float(
                validation_loss.item()
            )

            best_epoch = (
                epoch
            )

            best_state = {

                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for key, value
                in probe.state_dict().items()
            }

    # ========================================================
    # RESTORE BEST
    # ========================================================

    probe.load_state_dict(
        best_state
    )

    probe.eval()

    with torch.no_grad():

        validation_logits = (
            probe(
                validation_x
            )
        )

        validation_prediction = (
            validation_logits.argmax(
                dim=1
            )
        )

        final_validation_accuracy = float(
            (
                validation_prediction
                ==
                validation_y
            )
            .float()
            .mean()
            .item()
        )

    return {

        "probe":
            probe,

        "best_epoch":
            best_epoch,

        "best_validation_accuracy":
            final_validation_accuracy,

        "best_validation_loss":
            best_validation_loss,

        "history":
            history,

        "validation_prediction":
            validation_prediction
            .detach()
            .cpu()
    }


# ============================================================
# RUN PROBE FOR ONE MODEL
# ============================================================

def run_probe_experiment(
    model_name,
    model,
    train_loader,
    validation_loader,
    device,
    probe_seeds,
    probe_epochs,
    probe_lr
):

    print()
    print(
        "=============================================="
    )

    print(
        f"DOMAIN PROBE: {model_name}"
    )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # EXTRACT FROZEN EMBEDDINGS
    # --------------------------------------------------------

    train_data = extract_embeddings(
        model=model,
        loader=train_loader,
        device=device,
        label=
            f"{model_name} / train"
    )

    validation_data = extract_embeddings(
        model=model,
        loader=validation_loader,
        device=device,
        label=
            f"{model_name} / validation"
    )

    print()
    print(
        "Train embedding shape:",
        tuple(
            train_data[
                "embeddings"
            ].shape
        )
    )

    print(
        "Validation embedding shape:",
        tuple(
            validation_data[
                "embeddings"
            ].shape
        )
    )

    seed_results = []

    prediction_rows = []

    # ========================================================
    # MULTIPLE PROBE SEEDS
    # ========================================================

    for seed in probe_seeds:

        result = train_domain_probe(

            train_embeddings=
                train_data[
                    "embeddings"
                ],

            train_labels=
                train_data[
                    "domain_labels"
                ],

            validation_embeddings=
                validation_data[
                    "embeddings"
                ],

            validation_labels=
                validation_data[
                    "domain_labels"
                ],

            device=
                device,

            seed=
                seed,

            epochs=
                probe_epochs,

            lr=
                probe_lr
        )

        accuracy = (
            result[
                "best_validation_accuracy"
            ]
        )

        print()
        print(
            f"Probe seed {seed}"
        )

        print(
            "Best epoch:",
            result[
                "best_epoch"
            ]
        )

        print(
            "Validation domain accuracy:",
            f"{accuracy * 100:.2f}%"
        )

        seed_results.append({

            "model":
                model_name,

            "probe_seed":
                seed,

            "best_epoch":
                result[
                    "best_epoch"
                ],

            "validation_domain_accuracy":
                accuracy,

            "validation_domain_loss":
                result[
                    "best_validation_loss"
                ]
        })

        predictions = (
            result[
                "validation_prediction"
            ]
            .numpy()
        )

        labels = (
            validation_data[
                "domain_labels"
            ]
            .numpy()
        )

        for i in range(
            len(labels)
        ):

            prediction_rows.append({

                "model":
                    model_name,

                "probe_seed":
                    seed,

                "file_id":
                    validation_data[
                        "file_ids"
                    ][i],

                "source_file_id":
                    validation_data[
                        "source_ids"
                    ][i],

                "domain_label":
                    int(
                        labels[i]
                    ),

                "domain_prediction":
                    int(
                        predictions[i]
                    )
            })

    result_df = pd.DataFrame(
        seed_results
    )

    mean_accuracy = float(
        result_df[
            "validation_domain_accuracy"
        ].mean()
    )

    std_accuracy = float(
        result_df[
            "validation_domain_accuracy"
        ].std(
            ddof=0
        )
    )

    print()
    print(
        f"{model_name} mean probe accuracy:"
    )

    print(
        f"{mean_accuracy * 100:.2f}% "
        f"+/- "
        f"{std_accuracy * 100:.2f}%"
    )

    return {

        "seed_results":
            seed_results,

        "prediction_rows":
            prediction_rows,

        "mean_accuracy":
            mean_accuracy,

        "std_accuracy":
            std_accuracy
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--probe-lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--probe-seeds",
        nargs="+",
        type=int,
        default=[
            2026,
            2027,
            2028
        ]
    )

    args = parser.parse_args()

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
        "WEEK 9 - DOMAIN PROBE"
    )

    print(
        "=============================================="
    )

    print(
        "Device:",
        device
    )

    print(
        "Probe epochs:",
        args.probe_epochs
    )

    print(
        "Probe seeds:",
        args.probe_seeds
    )

    print(
        "Random chance:",
        f"{100 / N_DOMAIN:.2f}%"
    )

    # ========================================================
    # CHECK FILES
    # ========================================================

    required_files = [

        MANIFEST_PATH,
        NO_GRL_CHECKPOINT,
        WITH_GRL_CHECKPOINT,
        STATS_PATH
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    df = pd.read_csv(
        MANIFEST_PATH
    )

    # --------------------------------------------------------
    # DOMAIN SAMPLE ONLY
    # --------------------------------------------------------

    domain_df = df[
        df[
            "domain_label"
        ]
        >= 0
    ].copy()

    train_df = domain_df[
        domain_df[
            "split"
        ]
        ==
        "train"
    ].copy()

    validation_df = domain_df[
        domain_df[
            "split"
        ]
        ==
        "validation"
    ].copy()

    print()
    print(
        "Domain samples total:",
        len(
            domain_df
        )
    )

    print(
        "Train domain samples:",
        len(
            train_df
        )
    )

    print(
        "Validation domain samples:",
        len(
            validation_df
        )
    )

    # --------------------------------------------------------
    # EXPECTED
    # --------------------------------------------------------

    if len(
        train_df
    ) != 600:

        raise RuntimeError(
            "Train domain samples "
            f"seharusnya 600, "
            f"ditemukan {len(train_df)}."
        )

    if len(
        validation_df
    ) != 200:

        raise RuntimeError(
            "Validation domain samples "
            f"seharusnya 200, "
            f"ditemukan "
            f"{len(validation_df)}."
        )

    # ========================================================
    # DOMAIN DISTRIBUTION
    # ========================================================

    print()
    print(
        "Train domain distribution:"
    )

    print(
        train_df[
            "domain_label"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Validation domain distribution:"
    )

    print(
        validation_df[
            "domain_label"
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # VERIFY ALL 5 DOMAINS
    # --------------------------------------------------------

    train_domains = sorted(
        train_df[
            "domain_label"
        ]
        .unique()
        .tolist()
    )

    validation_domains = sorted(
        validation_df[
            "domain_label"
        ]
        .unique()
        .tolist()
    )

    expected_domains = [
        0,
        1,
        2,
        3,
        4
    ]

    if train_domains != expected_domains:

        raise RuntimeError(
            "Domain train tidak lengkap: "
            f"{train_domains}"
        )

    if validation_domains != expected_domains:

        raise RuntimeError(
            "Domain validation tidak lengkap: "
            f"{validation_domains}"
        )

    # ========================================================
    # SOURCE LEAKAGE CHECK
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

    leakage = (
        train_sources
        &
        validation_sources
    )

    if leakage:

        raise RuntimeError(
            f"Source leakage: "
            f"{len(leakage)}"
        )

    print()
    print(
        "Source leakage:",
        len(
            leakage
        )
    )

    # ========================================================
    # LOAD STATS
    # ========================================================

    stats = torch.load(
        STATS_PATH,
        map_location="cpu",
        weights_only=False
    )

    print()
    print(
        "Training feature statistics:"
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
    # DATASETS
    # ========================================================

    train_dataset = DomainProbeDataset(

        dataframe=
            train_df,

        feature_extractor=
            feature_extractor,

        stats=
            stats
    )

    validation_dataset = DomainProbeDataset(

        dataframe=
            validation_df,

        feature_extractor=
            feature_extractor,

        stats=
            stats
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=
            args.batch_size,

        shuffle=False,

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
    # LOAD BOTH MODELS
    # ========================================================

    no_grl_model, no_grl_checkpoint = (
        load_replay_model(
            NO_GRL_CHECKPOINT,
            device
        )
    )

    with_grl_model, with_grl_checkpoint = (
        load_replay_model(
            WITH_GRL_CHECKPOINT,
            device
        )
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
        "Validation EER:",
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
        "Validation EER:",
        f"{with_grl_checkpoint['validation_eer'] * 100:.2f}%"
    )

    # ========================================================
    # PROBE NO GRL
    # ========================================================

    no_grl_result = run_probe_experiment(

        model_name=
            "no_grl",

        model=
            no_grl_model,

        train_loader=
            train_loader,

        validation_loader=
            validation_loader,

        device=
            device,

        probe_seeds=
            args.probe_seeds,

        probe_epochs=
            args.probe_epochs,

        probe_lr=
            args.probe_lr
    )

    # ========================================================
    # PROBE WITH GRL
    # ========================================================

    with_grl_result = run_probe_experiment(

        model_name=
            "with_grl",

        model=
            with_grl_model,

        train_loader=
            train_loader,

        validation_loader=
            validation_loader,

        device=
            device,

        probe_seeds=
            args.probe_seeds,

        probe_epochs=
            args.probe_epochs,

        probe_lr=
            args.probe_lr
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = (
        RESULT_DIR
        /
        "domain_probe_summary.csv"
    )

    seed_path = (
        RESULT_DIR
        /
        "domain_probe_seeds.csv"
    )

    prediction_path = (
        RESULT_DIR
        /
        "domain_probe_predictions.csv"
    )

    config_path = (
        RESULT_DIR
        /
        "domain_probe_config.json"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = pd.DataFrame([

        {

            "model":
                "no_grl",

            "checkpoint_epoch":
                no_grl_checkpoint[
                    "epoch"
                ],

            "classification_validation_eer":
                no_grl_checkpoint[
                    "validation_eer"
                ],

            "domain_probe_accuracy_mean":
                no_grl_result[
                    "mean_accuracy"
                ],

            "domain_probe_accuracy_std":
                no_grl_result[
                    "std_accuracy"
                ]
        },

        {

            "model":
                "with_grl",

            "checkpoint_epoch":
                with_grl_checkpoint[
                    "epoch"
                ],

            "classification_validation_eer":
                with_grl_checkpoint[
                    "validation_eer"
                ],

            "domain_probe_accuracy_mean":
                with_grl_result[
                    "mean_accuracy"
                ],

            "domain_probe_accuracy_std":
                with_grl_result[
                    "std_accuracy"
                ]
        }
    ])

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # SEED RESULTS
    # --------------------------------------------------------

    seed_results = (

        no_grl_result[
            "seed_results"
        ]

        +

        with_grl_result[
            "seed_results"
        ]
    )

    pd.DataFrame(
        seed_results
    ).to_csv(
        seed_path,
        index=False
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    prediction_rows = (

        no_grl_result[
            "prediction_rows"
        ]

        +

        with_grl_result[
            "prediction_rows"
        ]
    )

    pd.DataFrame(
        prediction_rows
    ).to_csv(
        prediction_path,
        index=False
    )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = {

        "probe_type":
            "linear",

        "embedding_dim":
            EMBEDDING_DIM,

        "n_domain":
            N_DOMAIN,

        "random_chance":
            1.0 / N_DOMAIN,

        "train_domain_samples":
            len(
                train_df
            ),

        "validation_domain_samples":
            len(
                validation_df
            ),

        "probe_epochs":
            args.probe_epochs,

        "probe_learning_rate":
            args.probe_lr,

        "probe_seeds":
            args.probe_seeds,

        "no_grl_checkpoint":
            str(
                NO_GRL_CHECKPOINT
            ),

        "with_grl_checkpoint":
            str(
                WITH_GRL_CHECKPOINT
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
    # FINAL COMPARISON
    # ========================================================

    no_grl_accuracy = (
        no_grl_result[
            "mean_accuracy"
        ]
    )

    with_grl_accuracy = (
        with_grl_result[
            "mean_accuracy"
        ]
    )

    difference = (
        no_grl_accuracy
        -
        with_grl_accuracy
    )

    print()
    print(
        "=============================================="
    )

    print(
        "DOMAIN PROBE SELESAI"
    )

    print(
        "=============================================="
    )

    print()
    print(
        "Random chance:"
    )

    print(
        f"{100 / N_DOMAIN:.2f}%"
    )

    print()
    print(
        "NO GRL:"
    )

    print(
        f"{no_grl_accuracy * 100:.2f}% "
        f"+/- "
        f"{no_grl_result['std_accuracy'] * 100:.2f}%"
    )

    print()
    print(
        "WITH GRL:"
    )

    print(
        f"{with_grl_accuracy * 100:.2f}% "
        f"+/- "
        f"{with_grl_result['std_accuracy'] * 100:.2f}%"
    )

    print()
    print(
        "Penurunan domain probe accuracy:"
    )

    print(
        f"{difference * 100:.2f} percentage points"
    )

    print()
    print(
        "Classification validation EER:"
    )

    print(
        "NO GRL :",
        f"{no_grl_checkpoint['validation_eer'] * 100:.2f}%"
    )

    print(
        "WITH GRL:",
        f"{with_grl_checkpoint['validation_eer'] * 100:.2f}%"
    )

    print()
    print(
        "Output:"
    )

    print(
        summary_path
    )

    print(
        seed_path
    )

    print(
        prediction_path
    )

    print(
        config_path
    )

    print()

    if (
        with_grl_accuracy
        <
        no_grl_accuracy
    ):

        print(
            "HASIL SEMENTARA:"
        )

        print(
            "Domain probe accuracy lebih rendah "
            "pada embedding WITH GRL."
        )

        print(
            "Ini konsisten dengan berkurangnya "
            "informasi domain pada embedding."
        )

        print(
            "Klaim akhir tetap harus dikonfirmasi "
            "dengan unseen-factor evaluation."
        )

    elif (
        with_grl_accuracy
        >
        no_grl_accuracy
    ):

        print(
            "HASIL SEMENTARA:"
        )

        print(
            "Domain probe accuracy WITH GRL "
            "tidak lebih rendah."
        )

        print(
            "Belum ada bukti bahwa GRL "
            "mengurangi informasi domain."
        )

    else:

        print(
            "HASIL SEMENTARA:"
        )

        print(
            "Domain probe accuracy sama."
        )

        print(
            "Belum terlihat keuntungan GRL "
            "pada domain probing."
        )

if __name__ == "__main__":
    main()