from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly


# =========================================================
# KONFIGURASI
# =========================================================
TARGET_SR = 16000
TARGET_DURATION_SEC = 4.0
TARGET_NUM_SAMPLES = int(TARGET_SR * TARGET_DURATION_SEC)

N_FFT = 512
HOP_LENGTH = 160
N_MELS = 80

EPS = 1e-6


# =========================================================
# LOAD AUDIO
# =========================================================
def load_audio_fixed_4s(path, target_sr=TARGET_SR, target_len=TARGET_NUM_SAMPLES):
    """
    Load audio, convert to mono, resample ke 16 kHz,
    lalu trim / pad menjadi 4 detik.
    Output: tensor [1, 1, T]
    """
    audio, sr = sf.read(str(path), always_2d=True)

    # stereo -> mono
    audio = audio.mean(axis=1)

    # resample jika perlu
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr)

    # trim / pad
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        pad_width = target_len - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")

    wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return wav


# =========================================================
# SHARED STFT + MEL FILTERBANK
# =========================================================
class SpectralPhaseExtractor(torch.nn.Module):
    def __init__(self, sr=TARGET_SR, n_fft=N_FFT, hop=HOP_LENGTH, n_mels=N_MELS):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.n_mels = n_mels

        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sr / 2,
            n_mels=n_mels,
            sample_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("fb", fb.T)  # [mels, freq]
        self.window = torch.hann_window(n_fft)

    def shared_stft(self, wav):
        """
        wav: [B, 1, T]
        output X: [B, F, T_frames] complex
        """
        window = self.window.to(wav.device)

        X = torch.stft(
            wav.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
            center=False,
        )
        return X

    def compute_logmel(self, X):
        mag = X.abs().clamp_min(EPS)
        mel = torch.einsum("mf,bft->bmt", self.fb, mag)
        logmel = torch.log(mel + EPS)
        return logmel

    def compute_phase_derived(self, X):
        """
        Phase-derived representation:
        unwrap phase pada dimensi waktu,
        lalu hitung temporal phase difference,
        kemudian diproyeksikan ke Mel.
        """
        phase = torch.angle(X)  # [B, F, T]

        # unwrap pakai numpy agar kompatibel dan stabil
        phase_np = phase.detach().cpu().numpy()
        phase_unwrapped_np = np.unwrap(phase_np, axis=-1)
        phase_unwrapped = torch.from_numpy(phase_unwrapped_np).to(X.device)

        dphase = torch.diff(
            phase_unwrapped,
            dim=-1,
            prepend=phase_unwrapped[..., :1]
        ).abs()

        phase_mel = torch.einsum("mf,bft->bmt", self.fb, dphase)
        return phase_mel


# =========================================================
# MODIFIED GROUP DELAY
# =========================================================
def compute_mgd_mel(
    wav,
    sr=TARGET_SR,
    n_fft=N_FFT,
    hop=HOP_LENGTH,
    n_mels=N_MELS,
    alpha=0.4,
    gamma=0.9,
):
    """
    Menghitung Modified Group Delay (MGD) kemudian diproyeksikan ke Mel.

    wav: [B, 1, T]
    output: [B, 80, frames]
    """
    device = wav.device
    window = torch.hann_window(n_fft, device=device)

    # mel filterbank
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=sr / 2,
        n_mels=n_mels,
        sample_rate=sr,
        norm="slaney",
        mel_scale="slaney",
    ).T.to(device)  # [mels, freq]

    x = wav.squeeze(1)  # [B, T]
    B, T = x.shape

    n = torch.arange(T, device=device, dtype=x.dtype).unsqueeze(0).expand(B, -1)
    x_n = x * n

    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=False,
    )  # [B, F, frames]

    Y = torch.stft(
        x_n,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=False,
    )  # [B, F, frames]

    # Group delay numerator
    numerator = X.real * Y.real + X.imag * Y.imag

    # Smoothed magnitude approximation
    mag = X.abs().clamp_min(EPS)
    denominator = mag.pow(2 * gamma).clamp_min(EPS)

    tau = numerator / denominator

    # Modified group delay
    mgd = torch.sign(tau) * torch.abs(tau).pow(alpha)

    # dibuat non-negatif agar visual lebih mudah dibaca
    mgd = torch.abs(mgd)

    # proyeksi ke Mel
    mgd_mel = torch.einsum("mf,bft->bmt", fb, mgd)

    # optional stabilization
    mgd_mel = torch.log1p(mgd_mel)

    return mgd_mel


# =========================================================
# PLOT TERPISAH
# =========================================================
def save_feature_figure(feature_2d, title, output_path, cmap="magma"):
    """
    feature_2d: [mels, frames]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.imshow(
        feature_2d,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap
    )
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Mel Bin")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-path",
        type=str,
        default="data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac",
        help="Path audio input"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Folder output gambar"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    output_dir = Path(args.output_dir)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio tidak ditemukan: {audio_path}")

    print("======================================")
    print("PLOT THREE-CHANNEL FEATURES")
    print("======================================")
    print("Audio :", audio_path)

    wav = load_audio_fixed_4s(audio_path)
    print("Waveform shape:", wav.shape)

    extractor = SpectralPhaseExtractor()
    X = extractor.shared_stft(wav)

    logmel = extractor.compute_logmel(X)          # [1, 80, frames]
    phase_mel = extractor.compute_phase_derived(X) # [1, 80, frames]
    mgd_mel = compute_mgd_mel(wav)                 # [1, 80, frames]

    print("Log-Mel shape :", logmel.shape)
    print("Phase shape   :", phase_mel.shape)
    print("MGD shape     :", mgd_mel.shape)

    # ubah ke numpy 2D
    logmel_np = logmel[0].detach().cpu().numpy()
    phase_np = phase_mel[0].detach().cpu().numpy()
    mgd_np = mgd_mel[0].detach().cpu().numpy()

    # simpan gambar terpisah
    logmel_path = output_dir / "logmel.png"
    phase_path = output_dir / "phase_derived.png"
    mgd_path = output_dir / "mgd.png"

    save_feature_figure(
        logmel_np,
        "Log-Mel Spectrogram",
        logmel_path,
        cmap="magma"
    )

    save_feature_figure(
        phase_np,
        "Phase-Derived Representation",
        phase_path,
        cmap="viridis"
    )

    save_feature_figure(
        mgd_np,
        "Modified Group Delay",
        mgd_path,
        cmap="plasma"
    )

    print("\nSaved figures:")
    print(logmel_path)
    print(phase_path)
    print(mgd_path)

    print("\nSELESAI.")


if __name__ == "__main__":
    main()