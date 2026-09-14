from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from scipy.signal import resample_poly
from spectral_phase import (
    SpectralPhase,
    modified_group_delay
)

TARGET_SR = 16000
SECONDS = 4

def load_audio_4s(path):
    audio, sr = sf.read(
        path,
        always_2d=True
    )

    # stereo -> mono
    audio = audio.mean(axis=1)
    # resample
    if sr != TARGET_SR:
        audio = resample_poly(
            audio,
            TARGET_SR,
            sr
        )

    target = (
        TARGET_SR * SECONDS
    )

    # padding
    if len(audio) < target:
        audio = np.pad(
            audio,
            (0, target - len(audio))
        )
    # center crop
    elif len(audio) > target:
        start = (
            len(audio) - target
        ) // 2
        audio = audio[
            start:start + target
        ]

    # normalization
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
        .unsqueeze(0)
    )


def main():
    df = pd.read_csv(
        "manifests/source_manifest.csv"
    )
    row = None
    for _, r in df.iterrows():
        path = Path(
            r["file_path"]
        )
        if path.exists():
            row = r
            break

    if row is None:
        raise RuntimeError(
            "Tidak menemukan audio valid."
        )

    print(
        "Source ID:",
        row["source_file_id"]
    )

    print(
        "Audio:",
        row["file_path"]
    )

    # Load waveform
    wav = load_audio_4s(
        row["file_path"]
    )

    print(
        "\nWaveform:",
        wav.shape
    )

    # Feature extractor
    extractor = SpectralPhase()
    extractor.eval()

    with torch.no_grad():
        mgd = modified_group_delay(
            wav,
            extractor.fb,
            n_fft=512,
            hop=160,
            alpha=0.4,
            gamma=0.9
        )

    # Shape
    print(
        "\nMGD shape:",
        mgd.shape
    )

    # Numerical QC
    print(
        "\nNaN:",
        torch.isnan(
            mgd
        ).any().item()
    )

    print(
        "Inf:",
        torch.isinf(
            mgd
        ).any().item()
    )

    # Statistics
    print("\nMGD statistics:")
    print(
        "Min :",
        mgd.min().item()
    )

    print(
        "Max :",
        mgd.max().item()
    )

    print(
        "Mean:",
        mgd.mean().item()
    )

    print(
        "Std :",
        mgd.std().item()
    )

    flat = mgd.flatten()
    print(
        "P95 :",
        torch.quantile(
            flat,
            0.95
        ).item()
    )

    print(
        "P99 :",
        torch.quantile(
            flat,
            0.99
        ).item()
    )

    assert (
        mgd.shape[1] == 80
    ), "MGD bukan 80 Mel bins"

    assert not torch.isnan(
        mgd
    ).any(), "MGD menghasilkan NaN"

    assert not torch.isinf(
        mgd
    ).any(), "MGD menghasilkan Inf"

    print("\n====================================")
    print("TAHAP 5 - MODIFIED GROUP DELAY")
    print("====================================")
    print("MGD berhasil dihitung.")

if __name__ == "__main__":
    main()