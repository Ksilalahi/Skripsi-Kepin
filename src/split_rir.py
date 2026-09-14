from pathlib import Path
import shutil

source_dir = Path("data/rir/original")
seen_dir = Path("data/rir/seen")
unseen_dir = Path("data/rir/unseen")

seen_dir.mkdir(parents=True, exist_ok=True)
unseen_dir.mkdir(parents=True, exist_ok=True)

SEEN_IDENTITIES = {
    "booth",
    "lecture",
    "meeting",
    "office",
}

UNSEEN_IDENTITIES = {
    "phone",
    "stairway",
}

# Cari RIR
files = sorted(source_dir.rglob("*.wav"))
print("=" * 60)
print("SPLIT AIR DATABASE RIR")
print("=" * 60)
print("Total RIR ditemukan :", len(files))

seen_count = 0
unseen_count = 0
unknown_count = 0

# Proses
for file in files:
    parts = file.stem.split("_")

    if len(parts) < 2:
        print("UNKNOWN:", file.name)
        unknown_count += 1
        continue
    identity = parts[1].lower()

    # Seen
    if identity in SEEN_IDENTITIES:
        destination = seen_dir / file.name
        shutil.copy2(file, destination)
        seen_count += 1

    # Unseen
    elif identity in UNSEEN_IDENTITIES:
        destination = unseen_dir / file.name
        shutil.copy2(file, destination)
        unseen_count += 1
    else:
        print(
            f"Identity tidak dikenal: "
            f"{identity} -> {file.name}"
        )
        unknown_count += 1

print()
print("=" * 60)
print("HASIL SPLIT")
print("=" * 60)

print(f"Seen   : {seen_count}")
print(f"Unseen : {unseen_count}")
print(f"Unknown: {unknown_count}")

print()
print("Folder output:")
print("Seen   :", seen_dir)
print("Unseen :", unseen_dir)