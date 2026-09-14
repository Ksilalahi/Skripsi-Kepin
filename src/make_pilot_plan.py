import pandas as pd

INPUT = "manifests/acquisition_plan.csv"
OUTPUT = "manifests/pilot_acquisition_plan.csv"

df = pd.read_csv(INPUT)

# Ambil satu baris per source untuk pilot awal
pilot_base = (
    df.sort_values(["source_file_id", "repetition"])
      .groupby("source_file_id", as_index=False)
      .first()
)

# 10 bona fide
bona = pilot_base[
    pilot_base["label"] == 0
].head(10)

# 10 fake
fake = pilot_base[
    pilot_base["label"] == 1
].head(10)

pilot = pd.concat(
    [bona, fake],
    ignore_index=True
)

pilot["pilot_id"] = [
    f"PILOT_{i:03d}"
    for i in range(1, len(pilot) + 1)
]

pilot["output_path"] = (
    "data/real_replay/pilot/"
    + pilot["pilot_id"]
    + ".wav"
)

pilot["recording_status"] = "pending"

pilot.to_csv(
    OUTPUT,
    index=False
)

print("Jumlah pilot:", len(pilot))
print("\nDistribusi label:")
print(pilot["label"].value_counts().sort_index())

print("\nSaved:", OUTPUT)