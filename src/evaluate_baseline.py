from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch.utils.data import Dataset, DataLoader

from spectral_phase import SpectralPhase
from model_baseline import BaselineResNet
from metrics import compute_metrics

# CONFIG
TARGET_SR = 16000
SECONDS = 4
BATCH_SIZE = 4

MANIFEST = Path("manifests/source_manifest.csv")
CHECKPOINT_PATH = Path("checkpoints/baseline_resnet_clean.pt")
RESULTS_DIR = Path("results")

SCORES_PATH = RESULTS_DIR / "scores_baseline_clean.csv"
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

# AUDIO LOADER
def load_audio_4s(
    path,
    target_sr=TARGET_SR,
    seconds=SECONDS
):
    audio, sr = sf.read(
        str(path),
        always_2d=True
    )

    audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = resample_poly(
            audio,
            target_sr,
            sr
        )

    target = target_sr * seconds
    n = len(audio)
    if n < target:
        audio = np.pad(
            audio,
            (0, target - n)
        )

    elif n > target:
        start = (
            n - target
        ) // 2
        audio = audio[
            start:start + target
        ]

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-6:
        audio = audio / peak

    audio = audio.astype(
        np.float32
    )

    return (
        torch
        .from_numpy(audio)
        .unsqueeze(0)
    )

# DATASET
class CleanDataset(Dataset):
    def __init__(
        self,
        dataframe,
        extractor
    ):
        self.df = dataframe.reset_index(
            drop=True
        )

        self.extractor = extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = load_audio_4s(
            row["file_path"]
        )

        # [1, T] -> [1, 1, T]
        wav_batch = wav.unsqueeze(0)
        with torch.no_grad():
            logmel, _ = self.extractor(
                wav_batch
            )

        mean = logmel.mean()
        std = logmel.std().clamp_min(
            1e-6
        )

        logmel = (
            logmel - mean
        ) / std

        label = int(
            row["label"]
        )

        return (
            logmel,
            label,
            row["source_file_id"]
        )

# MAIN
def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # CEK CHECKPOINT
    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
    print("\nCheckpoint berhasil dibaca.")
    print(
        "Best epoch tersimpan:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    print(
        "Validation EER tersimpan:",
        checkpoint.get(
            "validation_eer",
            "unknown"
        )
    )

    print(
        "Seed:",
        checkpoint.get(
            "seed",
            "unknown"
        )
    )

    # MANIFEST
    df = pd.read_csv(
        MANIFEST
    )

    df = df[
        df["label"].isin(
            [0, 1]
        )
    ].copy()

    df = df[
        df["file_path"].apply(
            lambda p: Path(p).exists()
        )
    ].copy()

    # Test clean:
    # support nama split test/eval
    test_df = df[
        df["split"].isin(
            ["test", "eval"]
        )
    ].copy()
    print("\nJumlah test clean:",len(test_df))

    if len(test_df) == 0:
        raise RuntimeError("Test/eval split kosong.")
    print("\nDistribusi label test:")
    print(
        test_df[
            "label"
        ].value_counts()
        .sort_index()
    )

    # FEATURE EXTRACTOR
    extractor = (
        SpectralPhase()
        .cpu()
        .eval()
    )

    test_dataset = CleanDataset(
        test_df,
        extractor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # MODEL
    model = (
        BaselineResNet()
        .to(DEVICE)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # INFERENCE
    all_labels = []
    all_scores = []
    all_ids = []
    print("\nMulai evaluasi test clean...")

    with torch.no_grad():
        for batch_idx, (
            features,
            labels,
            source_ids
        ) in enumerate(test_loader):

            features = features.to(
                DEVICE
            )

            logits = model(
                features
            )

            probabilities = F.softmax(
                logits,
                dim=1
            )

            # class 1 = fake
            fake_score = probabilities[
                :,
                1
            ]
            all_labels.extend(
                labels.numpy()
            )
            all_scores.extend(
                fake_score
                .cpu()
                .numpy()
            )
            all_ids.extend(
                source_ids
            )
            if (
                (batch_idx + 1) % 50 == 0
            ):
                print("Processed batches:",batch_idx + 1)

    # METRICS
    metrics = compute_metrics(
        all_labels,
        all_scores
    )

    # SCORES CSV
    scores_df = pd.DataFrame({
        "file_id":
            all_ids,
        "source_file_id":
            all_ids,
        "label":
            all_labels,
        "fake_score":
            all_scores,
        "condition":
            "clean",
        "seed":
            checkpoint.get(
                "seed",
                2026
            )
    })
    scores_df.to_csv(
        SCORES_PATH,
        index=False
    )

    # RESULTS
    print("\n====================================")
    print("BASELINE RESNET - CLEAN TEST")
    print("====================================")
    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )
    val_eer = checkpoint.get(
        "validation_eer",
        None
    )

    if val_eer is not None:
        print("Validation EER:",f"{val_eer * 100:.2f}%")
    print("Test EER:",f"{metrics['eer'] * 100:.2f}%")
    print("EER threshold:",metrics["eer_threshold"])

    print("AUC:",f"{metrics['auc']:.4f}")
    print("F1:",f"{metrics['f1']:.4f}")
    print("Balanced Accuracy:",f"{metrics['balanced_accuracy']:.4f}")
    print("\nScores disimpan di:",SCORES_PATH)

if __name__ == "__main__":
    main()