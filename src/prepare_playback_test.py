from pathlib import Path

import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

MANIFEST = "manifests/source_manifest.csv"
OUTPUT = "data/real_replay/playback_master/test_playback.wav"
TARGET_SR = 48000

def main():
    df = pd.read_csv(MANIFEST)
    row = None
    for _, r in df.iterrows():
        path = Path(r["file_path"])
        if path.exists():
            row = r
            break

    if row is None:
        raise RuntimeError("Tidak menemukan file source yang valid.")

    input_path = Path(row["file_path"])
    output_path = Path(OUTPUT)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Source ID :", row["source_file_id"])
    print("Input     :", input_path)

    audio, sr = sf.read(
        input_path,
        always_2d=True
    )

    # mono
    audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = resample_poly(
            audio,
            TARGET_SR,
            sr
        )

    peak = abs(audio).max()
    if peak > 1e-6:
        audio = audio / peak * 0.8

    sf.write(
        output_path,
        audio,
        TARGET_SR,
        subtype="PCM_24"
    )

    print("Output    :", output_path)
    print("SampleRate:", TARGET_SR)
    print("Selesai.")

if __name__ == "__main__":
    main()