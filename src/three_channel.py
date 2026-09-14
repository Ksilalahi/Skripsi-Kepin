import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

from spectral_phase import (SpectralPhase,modified_group_delay)
from model_replay import ReplayResNet

TARGET_SR = 16000
SECONDS = 4

# LOAD AUDIO
def load_audio_4s(
    path,
    target_sr=TARGET_SR,
    seconds=SECONDS
):
    audio, sr = sf.read(
        str(path),
        always_2d=True
    )

    # stereo -> mono
    audio = audio.mean(axis=1)
    # resample
    if sr != target_sr:
        audio = resample_poly(
            audio,
            target_sr,
            sr
        )

    target = (
        target_sr * seconds
    )
    n = len(audio)
    # padding
    if n < target:
        audio = np.pad(
            audio,
            (0, target - n)
        )
    elif n > target:
        start = (n - target) // 2
        audio = audio[start:start + target]

    # peak normalization
    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-6:
        audio = audio / peak
    audio = audio.astype(
        np.float32
    )

    # [1,1,T]
    return (
        torch
        .from_numpy(audio)
        .unsqueeze(0)
        .unsqueeze(0)
    )

# MAIN
def main():
    df = pd.read_csv("manifests/source_manifest.csv")
    df = df[
        df["label"].isin([0, 1])
    ].copy()

    row = df.iloc[0]
    audio_path = row["file_path"]
    source_id = row["source_file_id"]

    print("Source ID:",source_id)
    print("Audio:",audio_path)

    # Load waveform
    wav = load_audio_4s(
        audio_path
    )
    print("\nWaveform:",wav.shape)

    # Spectral feature extractor
    extractor = (
        SpectralPhase()
        .eval()
    )

    with torch.no_grad():
        logmel, phase = extractor(
            wav
        )
        mgd = modified_group_delay(
            wav,
            extractor.fb,
            n_fft=512,
            hop=160,
            alpha=0.4,
            gamma=0.9
        )

    # Shape
    print("\nLog-Mel:",logmel.shape)
    print("Phase:",phase.shape)
    print("MGD:",mgd.shape)

    # Numerical QC
    print("\nNumerical check:")
    print(
        "Log-Mel NaN:",
        torch.isnan(
            logmel
        ).any().item()
    )
    print(
        "Phase NaN:",
        torch.isnan(
            phase
        ).any().item()
    )
    print(
        "MGD NaN:",
        torch.isnan(
            mgd
        ).any().item()
    )
    print(
        "MGD Inf:",
        torch.isinf(
            mgd
        ).any().item()
    )
    print("\nMGD statistics:")
    print("Min :",mgd.min().item())
    print("Max :",mgd.max().item())
    print("Mean:",mgd.mean().item())
    print("Std :",mgd.std().item())

    # Shape sama
    assert (
        logmel.shape
        ==
        phase.shape
        ==
        mgd.shape
    ), "Shape ketiga fitur tidak sama"

    def standardize(x):
        return (
            x - x.mean()
        ) / (
            x.std().clamp_min(
                1e-6
            )
        )

    logmel = standardize(logmel)
    phase = standardize(phase)
    mgd = standardize(mgd)

    # THREE CHANNEL
    features = torch.stack(
        [
            logmel,
            phase,
            mgd
        ],
        dim=1
    )

    print("\nThree-channel input:",features.shape)

    # Model
    model = ReplayResNet()
    model.eval()
    with torch.no_grad():
        class_logits, \
        transform_logits, \
        domain_logits, \
        embedding = model(
            features
        )

    # OUTPUT
    print("\nClass logits:",class_logits.shape)
    print("Transform logits:",transform_logits.shape)
    print("Domain logits    :", domain_logits.shape)
    print("Embedding:",embedding.shape)
    print("\n====================================")
    print("MINGGU 5 - THREE CHANNEL MODEL")
    print("====================================")
    print("Full feature berhasil dijalankan.")

if __name__ == "__main__":
    main()