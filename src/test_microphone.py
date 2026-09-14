import sounddevice as sd
import soundfile as sf
import numpy as np

SR = 48000
SECONDS = 5
INPUT_DEVICE = 1

print(
    "Mic:",
    sd.query_devices(INPUT_DEVICE)["name"]
)

print(
    "Mulai merekam 5 detik. Silakan bicara..."
)

recording = sd.rec(
    int(SECONDS * SR),
    samplerate=SR,
    channels=1,
    dtype="float32",
    device=INPUT_DEVICE
)

sd.wait()

peak = float(
    np.max(np.abs(recording))
)

rms = float(
    np.sqrt(np.mean(recording ** 2))
)

print("Peak:", peak)
print("RMS :", rms)

sf.write(
    "data/real_replay/pilot/test_mic_48000.wav",
    recording,
    SR,
    subtype="PCM_24"
)

print(
    "Saved: data/real_replay/pilot/test_mic_48000.wav"
)