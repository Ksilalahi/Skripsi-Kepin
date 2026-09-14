from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
import soundfile as sf

from scipy.signal import resample_poly
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ============================================================
# CONFIG
# ============================================================

TARGET_SR = 16000
DURATION_SEC = 4
TARGET_SAMPLES = TARGET_SR * DURATION_SEC

N_FFT = 512
HOP_LENGTH = 160
N_MELS = 80

EPS = 1e-6


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# BASELINE MODEL
# ============================================================

class CleanBaselineResNet(nn.Module):
    """
    ResNet18 khusus baseline clean:
    input = 1-channel Log-Mel
    output = 2 kelas:
        0 = bona fide
        1 = spoof/fake
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.net = resnet18(weights=None)

        # Checkpoint lama menunjukkan:
        # net.conv1.weight -> [64, 1, 7, 7]
        self.net.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.net.fc = nn.Linear(
            self.net.fc.in_features,
            num_classes,
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# AUDIO LOADER
# ============================================================

def load_audio_4s(path):
    """
    Evaluation:
    - mono
    - resample 16 kHz
    - pad jika < 4 s
    - center crop jika > 4 s
    - peak normalization
    """

    wav, sr = sf.read(
        str(path),
        always_2d=True,
        dtype="float32",
    )

    # [samples, channels] -> mono
    wav = wav.mean(axis=1)

    # Resample
    if sr != TARGET_SR:
        wav = resample_poly(
            wav,
            TARGET_SR,
            sr,
        ).astype(np.float32)

    n = len(wav)

    # Pad
    if n < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - n

        wav = np.pad(
            wav,
            (0, pad),
            mode="constant",
        )

    # Center crop
    elif n > TARGET_SAMPLES:
        start = (n - TARGET_SAMPLES) // 2

        wav = wav[
            start:start + TARGET_SAMPLES
        ]

    # Peak normalization
    peak = np.max(np.abs(wav))

    if peak > EPS:
        wav = wav / peak

    return torch.from_numpy(
        wav.astype(np.float32)
    )


# ============================================================
# MEL FILTERBANK
# ============================================================

def hz_to_mel(freq):
    return 2595.0 * np.log10(
        1.0 + freq / 700.0
    )


def mel_to_hz(mel):
    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def create_mel_filterbank(
    sr=TARGET_SR,
    n_fft=N_FFT,
    n_mels=N_MELS,
):

    n_freq = n_fft // 2 + 1

    mel_min = hz_to_mel(0.0)
    mel_max = hz_to_mel(sr / 2)

    mel_points = np.linspace(
        mel_min,
        mel_max,
        n_mels + 2,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    bins = np.floor(
        (n_fft + 1)
        * hz_points
        / sr
    ).astype(int)

    fb = np.zeros(
        (n_mels, n_freq),
        dtype=np.float32,
    )

    for m in range(1, n_mels + 1):

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        left = max(0, left)
        center = min(center, n_freq - 1)
        right = min(right, n_freq)

        if center > left:
            for k in range(left, center):
                fb[m - 1, k] = (
                    k - left
                ) / (
                    center - left
                )

        if right > center:
            for k in range(center, right):
                fb[m - 1, k] = (
                    right - k
                ) / (
                    right - center
                )

    return torch.from_numpy(fb)


MEL_FB = create_mel_filterbank()


# ============================================================
# LOG-MEL
# ============================================================

def extract_logmel(wav):
    """
    wav:
        [samples]

    output:
        [1, 80, frames]
    """

    window = torch.hann_window(
        N_FFT,
        device=wav.device,
    )

    X = torch.stft(
        wav,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=False,
        return_complex=True,
    )

    magnitude = X.abs()

    fb = MEL_FB.to(
        wav.device
    )

    mel = torch.matmul(
        fb,
        magnitude,
    )

    logmel = torch.log(
        mel + EPS
    )

    # ========================================================
    # PER-SAMPLE STANDARDIZATION
    # ========================================================
    #
    # Baseline lama kemungkinan menggunakan normalisasi
    # per-sample. Ini sengaja dibuat lokal hanya pada script
    # baseline dan TIDAK digunakan untuk pipeline akhir.
    #
    mean = logmel.mean()
    std = logmel.std().clamp_min(EPS)

    logmel = (
        logmel - mean
    ) / std

    # [80, T] -> [1, 80, T]
    return logmel.unsqueeze(0)


# ============================================================
# EER
# ============================================================

def compute_eer(
    y_true,
    scores,
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    scores = np.asarray(
        scores
    ).astype(float)

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    idx = np.nanargmin(
        np.abs(
            fpr - fnr
        )
    )

    eer = (
        fpr[idx]
        + fnr[idx]
    ) / 2.0

    threshold = thresholds[idx]

    return (
        float(eer),
        float(threshold),
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true,
    scores,
    fixed_threshold=None,
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    scores = np.asarray(
        scores
    ).astype(float)

    eer, eer_threshold = compute_eer(
        y_true,
        scores,
    )

    auc = roc_auc_score(
        y_true,
        scores,
    )

    threshold = (
        eer_threshold
        if fixed_threshold is None
        else fixed_threshold
    )

    pred = (
        scores >= threshold
    ).astype(int)

    f1 = f1_score(
        y_true,
        pred,
        zero_division=0,
    )

    bacc = balanced_accuracy_score(
        y_true,
        pred,
    )

    return {
        "eer": eer,
        "eer_percent": eer * 100.0,
        "eer_threshold": eer_threshold,
        "fixed_threshold": threshold,
        "auc": auc,
        "f1": f1,
        "balanced_accuracy": bacc,
    }


# ============================================================
# PATH RESOLVER
# ============================================================

def resolve_audio_path(
    file_path,
    project_root,
):

    path = Path(
        str(file_path)
    )

    # absolute
    if path.is_absolute():
        return path

    # relative terhadap project root
    candidate = (
        project_root / path
    )

    if candidate.exists():
        return candidate

    return path


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_split(
    model,
    dataframe,
    condition,
    device,
    seed,
    project_root,
):

    rows = []

    missing = 0
    failed = 0

    model.eval()

    total = len(dataframe)

    for i, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):

        source_id = str(
            row["source_file_id"]
        )

        audio_path = resolve_audio_path(
            row["file_path"],
            project_root,
        )

        if not audio_path.exists():

            print(
                f"[MISSING] {audio_path}"
            )

            missing += 1
            continue

        try:

            wav = load_audio_4s(
                audio_path
            ).to(device)

            feature = extract_logmel(
                wav
            )

            # [1, 80, T]
            # menjadi
            # [B=1, C=1, 80, T]

            feature = (
                feature
                .unsqueeze(0)
                .to(device)
            )

            logits = model(
                feature
            )

            # fake = class 1
            fake_score = (
                torch.softmax(
                    logits,
                    dim=1,
                )[0, 1]
                .item()
            )

            rows.append({
                "file_id": source_id,
                "source_file_id": source_id,
                "label": int(
                    row["label"]
                ),
                "fake_score": fake_score,
                "condition": condition,
                "seed": seed,
                "model": "Clean Baseline ResNet",
                "feature_set": "logmel",
            })

        except Exception as e:

            print(
                f"[ERROR] {source_id}: {e}"
            )

            failed += 1

        if (
            i % 100 == 0
            or i == total
        ):

            print(
                f"{condition}: "
                f"{i}/{total}"
            )

    print()
    print(
        f"{condition} selesai:"
    )

    print(
        "  Valid :",
        len(rows)
    )

    print(
        "  Missing:",
        missing
    )

    print(
        "  Failed :",
        failed
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="manifests/source_manifest.csv",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/baseline_resnet_clean.pt",
    )

    parser.add_argument(
        "--scores-out",
        default="results/scores_clean_baseline.csv",
    )

    parser.add_argument(
        "--metrics-out",
        default="results/metrics_clean_baseline.csv",
    )

    args = parser.parse_args()

    project_root = Path.cwd()

    checkpoint_path = Path(
        args.checkpoint
    )

    manifest_path = Path(
        args.manifest
    )

    scores_out = Path(
        args.scores_out
    )

    metrics_out = Path(
        args.metrics_out
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    scores_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    seed = int(
        checkpoint.get(
            "seed",
            2026,
        )
    )

    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "=========================================="
    )

    print(
        "CLEAN BASELINE EVALUATION"
    )

    print(
        "=========================================="
    )

    print(
        "Device     :",
        device
    )

    print(
        "Seed       :",
        seed
    )

    print(
        "Checkpoint :",
        checkpoint_path
    )

    print(
        "Epoch      :",
        checkpoint.get(
            "epoch"
        )
    )

    print(
        "Stored validation EER:",
        checkpoint.get(
            "validation_eer"
        )
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = CleanBaselineResNet(
        num_classes=2
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(
        device
    )

    print()
    print(
        "Checkpoint loaded successfully."
    )

    print(
        "Input channel:",
        model.net.conv1.in_channels,
    )

    print(
        "Output classes:",
        model.net.fc.out_features,
    )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    df = pd.read_csv(
        manifest_path
    )

    required = {
        "source_file_id",
        "file_path",
        "label",
        "split",
    }

    missing_columns = (
        required
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest tidak memiliki kolom: "
            + str(missing_columns)
        )

    # pastikan label valid
    df = df[
        df["label"].isin(
            [0, 1]
        )
    ].copy()

    split_lower = (
        df["split"]
        .astype(str)
        .str.lower()
    )

    validation_df = df[
        split_lower.isin(
            ["dev", "validation", "val"]
        )
    ].copy()

    test_df = df[
        split_lower.isin(
            ["eval", "test"]
        )
    ].copy()

    print()
    print(
        "Clean validation:",
        len(validation_df)
    )

    print(
        "Clean test      :",
        len(test_df)
    )

    if len(validation_df) == 0:
        raise RuntimeError(
            "Validation clean kosong."
        )

    if len(test_df) == 0:
        raise RuntimeError(
            "Test clean kosong."
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print(
        "Evaluating clean validation..."
    )

    val_scores = evaluate_split(
        model=model,
        dataframe=validation_df,
        condition="clean_validation",
        device=device,
        seed=seed,
        project_root=project_root,
    )

    if len(
        val_scores["label"].unique()
    ) < 2:

        raise RuntimeError(
            "Validation harus berisi label 0 dan 1."
        )

    val_metrics = compute_metrics(
        val_scores["label"],
        val_scores["fake_score"],
    )

    validation_threshold = (
        val_metrics[
            "eer_threshold"
        ]
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print()
    print(
        "Evaluating clean test..."
    )

    test_scores = evaluate_split(
        model=model,
        dataframe=test_df,
        condition="clean_test",
        device=device,
        seed=seed,
        project_root=project_root,
    )

    if len(
        test_scores["label"].unique()
    ) < 2:

        raise RuntimeError(
            "Test harus berisi label 0 dan 1."
        )

    test_metrics = compute_metrics(
        test_scores["label"],
        test_scores["fake_score"],
        fixed_threshold=validation_threshold,
    )

    # --------------------------------------------------------
    # SAVE SCORES
    # --------------------------------------------------------

    all_scores = pd.concat(
        [
            val_scores,
            test_scores,
        ],
        ignore_index=True,
    )

    all_scores.to_csv(
        scores_out,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_rows = [
        {
            "condition": "clean_validation",
            "seed": seed,
            "model": "Clean Baseline ResNet",
            "feature_set": "logmel",
            "eer": val_metrics["eer"],
            "eer_percent": val_metrics[
                "eer_percent"
            ],
            "eer_threshold": val_metrics[
                "eer_threshold"
            ],
            "fixed_threshold": val_metrics[
                "fixed_threshold"
            ],
            "auc": val_metrics["auc"],
            "f1": val_metrics["f1"],
            "balanced_accuracy":
                val_metrics[
                    "balanced_accuracy"
                ],
            "n_samples": len(
                val_scores
            ),
        },
        {
            "condition": "clean_test",
            "seed": seed,
            "model": "Clean Baseline ResNet",
            "feature_set": "logmel",
            "eer": test_metrics["eer"],
            "eer_percent": test_metrics[
                "eer_percent"
            ],
            "eer_threshold": test_metrics[
                "eer_threshold"
            ],
            "fixed_threshold":
                validation_threshold,
            "auc": test_metrics["auc"],
            "f1": test_metrics["f1"],
            "balanced_accuracy":
                test_metrics[
                    "balanced_accuracy"
                ],
            "n_samples": len(
                test_scores
            ),
        },
    ]

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_df.to_csv(
        metrics_out,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL PRINT
    # --------------------------------------------------------

    print()
    print(
        "=========================================="
    )

    print(
        "CLEAN BASELINE RESULT"
    )

    print(
        "=========================================="
    )

    print()
    print(
        "[VALIDATION]"
    )

    print(
        f"EER      : "
        f"{val_metrics['eer_percent']:.2f}%"
    )

    print(
        f"Threshold: "
        f"{val_metrics['eer_threshold']:.8f}"
    )

    print(
        f"AUC      : "
        f"{val_metrics['auc']:.4f}"
    )

    print()
    print(
        "[CLEAN TEST]"
    )

    print(
        f"EER      : "
        f"{test_metrics['eer_percent']:.2f}%"
    )

    print(
        f"EER thr  : "
        f"{test_metrics['eer_threshold']:.8f}"
    )

    print(
        f"Val thr  : "
        f"{validation_threshold:.8f}"
    )

    print(
        f"AUC      : "
        f"{test_metrics['auc']:.4f}"
    )

    print(
        f"F1       : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Bal Acc  : "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        scores_out
    )

    print(
        metrics_out
    )

    print()
    print(
        "results/scores.csv TIDAK diubah."
    )

if __name__ == "__main__":
    main()