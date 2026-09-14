import pandas as pd

manifest_path = "manifests/source_manifest.csv"

df = pd.read_csv(manifest_path)

print("=" * 50)
print("PEMERIKSAAN SOURCE MANIFEST")
print("=" * 50)

print(f"Total file audio : {len(df)}")

print("\nJumlah berdasarkan label:")

count = df["label"].value_counts().sort_index()

print(f"Bona fide (real) : {(df['label'] == 0).sum()}")
print(f"Spoof (fake)     : {(df['label'] == 1).sum()}")
print(f"Unknown          : {(df['label'] == -1).sum()}")

print("\nPersentase:")

total_valid = ((df["label"] == 0) | (df["label"] == 1)).sum()

if total_valid > 0:
    real = (df["label"] == 0).sum()
    fake = (df["label"] == 1).sum()

    print(f"Bona fide : {real / total_valid * 100:.2f}%")
    print(f"Spoof     : {fake / total_valid * 100:.2f}%")

print("\n" + "=" * 50)
print("JUMLAH BERDASARKAN SPLIT")
print("=" * 50)

split_table = pd.crosstab(
    df["split"],
    df["label"]
)

split_table = split_table.rename(
    columns={
        0: "bona_fide",
        1: "spoof",
        -1: "unknown"
    }
)

print(split_table)

print("\n" + "=" * 50)
print("JUMLAH BERDASARKAN ATTACK ID")
print("=" * 50)

attack_table = pd.crosstab(
    df["attack_id"],
    df["label"]
)

attack_table = attack_table.rename(
    columns={
        0: "bona_fide",
        1: "spoof"
    }
)

print(attack_table)