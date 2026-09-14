from pathlib import Path
import soundfile as sf

rir_dir = Path("data/rir/original")
files = sorted(rir_dir.rglob("*.wav"))

print("=" * 60)
print("AIR DATABASE - RIR CHECK")
print("=" * 60)
print(f"Jumlah file WAV : {len(files)}")

if len(files) == 0:
    print("\nTidak ada file .wav ditemukan.")
    print("Masukkan file RIR .wav ke:")
    print("data/rir/original/")
    raise SystemExit

print("\n10 file pertama:")
for file in files[:10]:
    print(file)
print("\nInformasi audio:")

for file in files[:5]:
    info = sf.info(str(file))

    print("-" * 60)
    print("File        :", file.name)
    print("Sample rate :", info.samplerate)
    print("Channels    :", info.channels)
    print("Frames      :", info.frames)
    print("Duration    :", round(info.duration, 4), "detik")