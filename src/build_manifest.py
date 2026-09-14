from pathlib import Path
import hashlib, soundfile as sf, pandas as pd

from pathlib import Path

def read_protocol(protocol_path, split_name):
    protocol = {}
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            speaker_id = parts[0]
            file_id = parts[1]
            attack_id = parts[3]
            label = parts[4]

            protocol[file_id] = {
                "speaker_id": speaker_id,
                "attack_id": attack_id,
                "label": 0 if label == "bonafide" else 1,
                "split": split_name
            }
    return protocol

protocol = {}
protocol.update(
    read_protocol(
        "data/raw/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "train"
    )
)
protocol.update(
    read_protocol(
        "data/raw/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "dev"
    )
)
protocol.update(
    read_protocol(
        "data/raw/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        "eval"
    )
)

print(f"Protocol loaded: {len(protocol)} entries")

def sha256(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()

def inspect_audio(path):
    info = sf.info(str(path))
    return info.frames / info.samplerate, info.samplerate, info.channels

rows = []
skipped = 0
for path in Path("data/raw").rglob("*.flac"):
    duration, sr, channels = inspect_audio(path)
    if duration < 1.0:
        continue

    file_id = path.stem
    meta = protocol.get(file_id)
    if meta is None:
        skipped += 1
        print(f"Skip : {file_id}")
        continue
    rows.append({
        "source_file_id": file_id,
        "file_path": str(path),
        "speaker_id": meta["speaker_id"],
        "attack_id": meta["attack_id"],
        "label": meta["label"],
        "dataset": "ASVspoof2019_LA",
        "split": meta["split"],
        "duration_s": round(duration, 4),
        "sample_rate": sr,
        "channels": channels,
        "sha256": sha256(path),

    })

pd.DataFrame(rows).to_csv("manifests/source_manifest.csv", index=False)
print("Rows:", len(rows))