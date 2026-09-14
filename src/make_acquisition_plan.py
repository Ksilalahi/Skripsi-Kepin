import pandas as pd
import numpy as np

src = pd.read_csv("manifests/source_manifest.csv")
src = src.groupby("label", group_keys=False).sample(n=60, random_state=2026)
configs = [
    {"room":"A", "playback":"speaker_A", "mic":"mic_A", "distance":0.5, "angle":0},
    {"room":"A", "playback":"speaker_B", "mic":"mic_B", "distance":1.0, "angle":45},
    {"room":"B", "playback":"speaker_A", "mic":"mic_unseen", "distance":0.5, "angle":45},
    {"room":"B", "playback":"speaker_B", "mic":"mic_unseen", "distance":1.0, "angle":0},
]
rows = []
for i, row in src.reset_index(drop=True).iterrows():
    chosen = [configs[i % len(configs)], configs[(i + 1) % len(configs)]]
    for c in chosen:
        for rep in (1, 2):
            rows.append({**row.to_dict(), **c, "repetition":rep})
pd.DataFrame(rows).to_csv("manifests/acquisition_plan.csv", index=False)