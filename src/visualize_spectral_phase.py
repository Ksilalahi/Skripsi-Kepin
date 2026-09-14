import torch
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from pathlib import Path

from spectral_phase import SpectralPhase

TARGET_SR = 16000
OUTPUT_DIR = Path("results/week3_visual")

def load_audio(path, target_sr=TARGET_SR):
    audio, sr = sf.read(path)

    # stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # resample
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr)

    audio = audio.astype(np.float32)

    # normalisasi peak
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak

    wav = torch.from_numpy(audio)
    return wav.unsqueeze(0).unsqueeze(0)  # [1, 1, T]

def save_feature_image(feature, title, out_path):
    feat = feature.squeeze(0).cpu().numpy()  # [80, frames]
    plt.figure(figsize=(10, 4))
    plt.imshow(feat, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Mel Bin")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    source_manifest = pd.read_csv("manifests/source_manifest.csv")
    replay_manifest = pd.read_csv("manifests/manifest_replay_sim.csv")

    # Ambil 1 contoh replay
    replay_row = replay_manifest.iloc[0]
    replay_path = replay_row["file_path"]
    source_id = replay_row["source_file_id"]

    # Cari clean source yang sama
    clean_row = source_manifest[
        source_manifest["source_file_id"] == source_id
    ].iloc[0]
    clean_path = clean_row["file_path"]

    print("Source ID   :", source_id)
    print("Clean path  :", clean_path)
    print("Replay path :", replay_path)

    # Load audio
    clean_wav = load_audio(clean_path)
    replay_wav = load_audio(replay_path)

    print("\nAudio tensor shape:")
    print("Clean :", clean_wav.shape)
    print("Replay:", replay_wav.shape)

    # Feature extraction
    extractor = SpectralPhase()

    with torch.no_grad():
        clean_logmel, clean_phase = extractor(clean_wav)
        replay_logmel, replay_phase = extractor(replay_wav)

    # Print shape
    print("\nFeature shape:")
    print("Clean Log-Mel :", clean_logmel.shape)
    print("Clean Phase   :", clean_phase.shape)
    print("Replay Log-Mel:", replay_logmel.shape)
    print("Replay Phase  :", replay_phase.shape)

    # NaN check
    print("\nNaN check:")
    print("Clean Log-Mel :", torch.isnan(clean_logmel).any().item())
    print("Clean Phase   :", torch.isnan(clean_phase).any().item())
    print("Replay Log-Mel:", torch.isnan(replay_logmel).any().item())
    print("Replay Phase  :", torch.isnan(replay_phase).any().item())

    # Save images
    save_feature_image(
        clean_logmel,
        "Clean - Log-Mel",
        OUTPUT_DIR / "clean_logmel.png"
    )

    save_feature_image(
        clean_phase,
        "Clean - Phase Map",
        OUTPUT_DIR / "clean_phase.png"
    )

    save_feature_image(
        replay_logmel,
        "Simulated Replay - Log-Mel",
        OUTPUT_DIR / "replay_logmel.png"
    )

    save_feature_image(
        replay_phase,
        "Simulated Replay - Phase Map",
        OUTPUT_DIR / "replay_phase.png"
    )

    print("\nVisual berhasil disimpan di:")
    print(OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()