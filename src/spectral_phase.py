import torch, torchaudio
import torch.nn.functional as F

class SpectralPhase(torch.nn.Module):
    def __init__(self, sr=16000, n_fft=512, hop=160, n_mels=80):
        super().__init__()
        self.sr, self.n_fft, self.hop = sr, n_fft, hop
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft//2 + 1, f_min=0.0, f_max=sr/2,
            n_mels=n_mels, sample_rate=sr, norm="slaney",
            mel_scale="slaney")
        self.register_buffer("fb", fb.T)  # [mels, freq]
        self.window = torch.hann_window(n_fft)

    def forward(self, wav):
        window = self.window.to(wav.device)
        X = torch.stft(wav.squeeze(1), n_fft=self.n_fft,
                       hop_length=self.hop, win_length=self.n_fft,
                       window=window, return_complex=True, center=False)
        mag = X.abs().clamp_min(1e-6)
        logmel = torch.log(torch.einsum("mf,bft->bmt", self.fb, mag) + 1e-6)
        phase = torch.angle(X)
        dphase = torch.diff(phase, dim=-1, prepend=phase[..., :1]).abs()
        phase_mel = torch.einsum("mf,bft->bmt", self.fb, dphase)
        return logmel, phase_mel

def modified_group_delay(wav, fb, n_fft=512, hop=160, alpha=0.4, gamma=0.9):
    b, _, t = wav.shape
    device = wav.device
    window = torch.hann_window(n_fft, device=device)
    X = torch.stft(wav.squeeze(1), n_fft=n_fft, hop_length=hop,
                   win_length=n_fft, window=window,
                   return_complex=True, center=False)
    n = torch.arange(t, device=device).float()[None, :]
    Y = torch.stft((wav.squeeze(1) * n), n_fft=n_fft, hop_length=hop,
                   win_length=n_fft, window=window,
                   return_complex=True, center=False)
    numerator = (X.real * Y.real + X.imag * Y.imag)
    mag = X.abs().clamp_min(1e-6)
    smooth = F.avg_pool2d(mag.unsqueeze(1), kernel_size=(5,1),
                          stride=1, padding=(2,0)).squeeze(1)
    tau = numerator / (smooth.pow(2 * gamma) + 1e-6)
    mgd = tau.sign() * tau.abs().clamp_min(1e-6).pow(alpha)
    return torch.einsum("mf,bft->bmt", fb, mgd.abs())
