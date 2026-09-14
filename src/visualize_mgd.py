from pathlib import Path

import matplotlib.pyplot as plt
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

def load_audio(path):
    audio, sr = sf.read(
        path,
        always_2d=True
    )

    audio = audio.mean(axis=1)
    if sr != TARGET_SR:

        audio = resample_poly(
            audio,
            TARGET_SR,
            sr
        )

    target = TARGET_SR * SECONDS

    if len(audio) < target:

        audio = np.pad(
            audio,
            (0, target - len(audio))
        )

    elif len(audio) > target:

        start = (
            len(audio) - target
        ) // 2

        audio = audio[
            start:start + target
        ]

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-6:

        audio = audio / peak

    return (
        torch.tensor(
            audio,
            dtype=torch.float32
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )


df = pd.read_csv(
    "manifests/source_manifest.csv"
)


row = None

for _, r in df.iterrows():

    if Path(
        r["file_path"]
    ).exists():

        row = r
        break


wav = load_audio(
    row["file_path"]
)


extractor = SpectralPhase()

extractor.eval()


with torch.no_grad():

    mgd = modified_group_delay(
        wav,
        extractor.fb
    )


mgd_np = (
    mgd[0]
    .cpu()
    .numpy()
)


plt.figure(
    figsize=(10, 4)
)

plt.imshow(
    mgd_np,
    origin="lower",
    aspect="auto"
)

plt.colorbar(
    label="MGD magnitude"
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Mel bin"
)

plt.title(
    f"Modified Group Delay - {row['source_file_id']}"
)

plt.tight_layout()

plt.show()