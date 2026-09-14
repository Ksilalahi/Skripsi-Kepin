from pathlib import Path
import soundfile as sf

folders = [
    Path("data/rir/seen_16k"),
    Path("data/rir/unseen_16k"),
]

for folder in folders:
    files = sorted(folder.rglob("*.wav"))
    print()
    print("=" * 60)
    print("Folder:", folder)
    print("Jumlah file:", len(files))
    print("=" * 60)

    if len(files) == 0:
        print("Tidak ada file WAV.")
        continue

    sample_rates = set()
    channels = set()

    for file in files:
        info = sf.info(str(file))
        sample_rates.add(info.samplerate)
        channels.add(info.channels)

    print("Sample rate:", sample_rates)
    print("Channels   :", channels)
    print("\nContoh file:")

    for file in files[:5]:
        info = sf.info(str(file))
        print(
            f"{file.name} | "
            f"SR={info.samplerate} | "
            f"Channels={info.channels} | "
            f"Duration={info.duration:.4f}s"
        )