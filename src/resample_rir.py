from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# KONFIGURASI
TARGET_SR = 16000
INPUT_SEEN = Path("data/rir/seen")
INPUT_UNSEEN = Path("data/rir/unseen")

OUTPUT_SEEN = Path("data/rir/seen_16k")
OUTPUT_UNSEEN = Path("data/rir/unseen_16k")

OUTPUT_SEEN.mkdir(parents=True, exist_ok=True)
OUTPUT_UNSEEN.mkdir(parents=True, exist_ok=True)

# FUNGSI RESAMPLING
def resample_audio(input_path, output_path):
    audio, sr = sf.read(
        str(input_path),
        always_2d=True
    )

    # Jika sudah 16 kHz
    if sr == TARGET_SR:
        resampled = audio
    else:
        # Resampling berdasarkan rasio sample rate
        gcd = np.gcd(sr, TARGET_SR)
        up = TARGET_SR // gcd
        down = sr // gcd
        channels = []
        for ch in range(audio.shape[1]):
            channel = audio[:, ch]
            channel_resampled = resample_poly(
                channel,
                up,
                down
            )
            channels.append(channel_resampled)
        resampled = np.stack(
            channels,
            axis=1
        )

    sf.write(
        str(output_path),
        resampled,
        TARGET_SR,
        subtype="PCM_24"
    )

def process_folder(input_dir, output_dir):
    files = sorted(input_dir.rglob("*.wav"))

    print()
    print("=" * 60)
    print(f"INPUT  : {input_dir}")
    print(f"OUTPUT : {output_dir}")
    print("=" * 60)
    print("Jumlah file :", len(files))

    success = 0
    skipped = 0
    failed = 0

    for i, input_file in enumerate(files, start=1):
        output_file = output_dir / input_file.name
        try:
            # Baca informasi terlebih dahulu
            info = sf.info(str(input_file))

            # Pastikan RIR tidak kosong
            if info.frames == 0:
                print(
                    f"[SKIP] File kosong: "
                    f"{input_file.name}"
                )
                skipped += 1
                continue
            resample_audio(
                input_file,
                output_file
            )
            success += 1
            print(
                f"[{i}/{len(files)}] "
                f"{input_file.name}"
            )

        except Exception as e:
            failed += 1
            print(
                f"[ERROR] "
                f"{input_file.name}: {e}"
            )

    print()
    print("Hasil:")
    print("Berhasil :", success)
    print("Skipped  :", skipped)
    print("Error    :", failed)


# MAIN
if __name__ == "__main__":
    print("=" * 60)
    print("RESAMPLING AIR DATABASE RIR")
    print("=" * 60)
    process_folder(
        INPUT_SEEN,
        OUTPUT_SEEN
    )

    process_folder(
        INPUT_UNSEEN,
        OUTPUT_UNSEEN
    )

    print()
    print("=" * 60)
    print("RESAMPLING SELESAI")
    print("=" * 60)