from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from scipy.signal import resample_poly
from spectral_phase import SpectralPhase
TARGET_SR = 16000
SECONDS = 4

def load_audio_4s(path):
    audio, sr = sf.read(
        path,
        always_2d=True
    )
    # stereo -> mono
    audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = resample_poly(
            audio,
            TARGET_SR,
            sr
        )

    target = (
        TARGET_SR * SECONDS
    )

    if len(audio) < target:
        audio = np.pad(
            audio,
            (0, target - len(audio))
        )
    elif len(audio) > target:
        start = (len(audio) - target) // 2
        audio = audio[
            start:start + target
        ]

    peak = np.max(np.abs(audio))
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
    df = pd.read_csv("manifests/source_manifest.csv")
    row = None

    for _, r in df.iterrows():
        if Path(
            r["file_path"]
        ).exists():
            row = r
            break

    if row is None:
        raise RuntimeError(
            "Tidak ada audio valid."
        )
    print("\n===================================================")
    print("Tahap 4 - Mengekstraksi log-Mel dan phase-derived map")
    print("=====================================================")
    print(
        "Source:",
        row["source_file_id"]
    )

    wav = load_audio_4s(
        row["file_path"]
    )

    print(
        "Waveform:",
        wav.shape
    )

    extractor = SpectralPhase()
    extractor.eval()

    with torch.no_grad():
        logmel, phase_map = extractor(wav)

    print("\nLog-Mel:")
    print(logmel.shape)

    print("\nPhase map:")
    print(phase_map.shape)


    # STACK 2 CHANNEL
    features = torch.stack(
        [
            logmel,
            phase_map
        ],
        dim=1
    )

    print("\nTwo-channel tensor:")
    print(features.shape)

    # QC
    print(
        "\nNaN Log-Mel:",
        torch.isnan(
            logmel
        ).any().item()
    )

    print(
        "NaN Phase:",
        torch.isnan(
            phase_map
        ).any().item()
    )

    print(
        "Inf Log-Mel:",
        torch.isinf(
            logmel
        ).any().item()
    )

    print(
        "Inf Phase:",
        torch.isinf(
            phase_map
        ).any().item()
    )

    assert (
        logmel.shape
        ==
        phase_map.shape
    ), "Ukuran Log-Mel dan phase map berbeda"


    assert not torch.isnan(
        features
    ).any(), "Terdapat NaN"

    assert not torch.isinf(
        features
    ).any(), "Terdapat Inf"

    print("Log-Mel dan phase-derived berhasil.")

if __name__ == "__main__":
    main()