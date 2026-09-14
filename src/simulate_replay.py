from pathlib import Path
import random

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 2026
TARGET_SR = 16000

SNR_LIST = [20, 10, 5]

SOURCE_MANIFEST = Path(
    "manifests/source_manifest.csv"
)

RIR_ROOT = Path(
    "data/rir/original"
)

# Jangan timpa 50 contoh Week 2
OUTPUT_DIR = Path(
    "data/processed/replay_sim_week9"
)

OUTPUT_MANIFEST = Path(
    "manifests/manifest_replay_sim_week9.csv"
)


# ============================================================
# TARGET DATASET WEEK 9
# ============================================================

TRAIN_PER_LABEL = 300
VALIDATION_PER_LABEL = 100
TEST_PER_LABEL = 100

# Total:
# train      = 600
# validation = 200
# test       = 200
# TOTAL      = 1000


# ============================================================
# RIR CONFIGURATION
# ============================================================

# Lima domain RIR yang digunakan sebagai SEEN.
SEEN_IDENTITIES = [
    "booth",
    "lecture",
    "meeting",
    "office",
    "phone"
]

# Stairway sepenuhnya dipisahkan sebagai unseen identity.
UNSEEN_IDENTITIES = [
    "stairway"
]

# Satu RIR dipilih dari masing-masing seen identity.
# Total = 5 RIR seen.
N_SEEN_RIR = 5

# Dua RIR berbeda dari stairway sebagai unseen.
N_UNSEEN_RIR = 2


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# AUDIO FUNCTIONS
# ============================================================

def load_audio(path, target_sr=TARGET_SR):

    audio, sr = sf.read(
        str(path),
        always_2d=True
    )

    # Mono
    audio = audio.mean(
        axis=1
    )

    # Resample
    if sr != target_sr:

        audio = resample_poly(
            audio,
            target_sr,
            sr
        )

    wav = torch.from_numpy(
        audio.astype(
            np.float32
        )
    ).unsqueeze(0)

    return wav


def apply_rir(wav, rir):

    # Normalisasi RIR
    rir = (
        rir
        /
        rir.abs()
        .max()
        .clamp_min(1e-6)
    )

    n = (
        wav.shape[-1]
        +
        rir.shape[-1]
        - 1
    )

    y = torch.fft.irfft(

        torch.fft.rfft(
            wav,
            n=n
        )
        *
        torch.fft.rfft(
            rir,
            n=n
        ),

        n=n
    )

    # Pertahankan panjang waveform sumber
    y = y[
        ...,
        :wav.shape[-1]
    ]

    return y


def add_noise_snr(
    wav,
    snr_db
):

    noise = torch.randn_like(
        wav
    )

    signal_rms = (
        wav.pow(2)
        .mean()
        .sqrt()
        .clamp_min(1e-6)
    )

    noise_rms = (
        noise.pow(2)
        .mean()
        .sqrt()
        .clamp_min(1e-6)
    )

    noise = noise * (
        signal_rms
        /
        (
            10 ** (snr_db / 20)
            *
            noise_rms
        )
    )

    return wav + noise


def normalize_peak(
    wav,
    target_peak=0.95
):

    peak = (
        wav.abs()
        .max()
        .clamp_min(1e-6)
    )

    return (
        wav
        /
        peak
        *
        target_peak
    )


# ============================================================
# RIR FUNCTIONS
# ============================================================

def get_rir_identity(path):

    name = path.stem.lower()

    identities = [
        "booth",
        "lecture",
        "meeting",
        "office",
        "phone",
        "stairway"
    ]

    for identity in identities:

        if identity in name:
            return identity

    return "unknown"


def build_rir_table():

    rir_files = sorted(
        RIR_ROOT.rglob(
            "*.wav"
        )
    )

    if len(rir_files) == 0:

        raise RuntimeError(
            f"Tidak ada file RIR di {RIR_ROOT}"
        )

    rows = []

    for path in rir_files:

        rows.append({
            "path": path,
            "identity": get_rir_identity(
                path
            )
        })

    return rows


def select_rirs(
    rir_table
):

    rng = random.Random(
        SEED
    )

    # --------------------------------------------------------
    # SEEN
    # --------------------------------------------------------
    #
    # Pilih SATU RIR dari setiap identity:
    #
    # booth
    # lecture
    # meeting
    # office
    # phone
    #
    # Dengan demikian benar-benar ada 5 seen domains.
    # --------------------------------------------------------

    seen_rirs = []

    for identity in SEEN_IDENTITIES:

        candidates = [
            row["path"]
            for row in rir_table
            if row["identity"] == identity
        ]

        if len(candidates) == 0:

            raise RuntimeError(
                f"Tidak ada RIR identity: "
                f"{identity}"
            )

        selected = rng.choice(
            candidates
        )

        seen_rirs.append(
            selected
        )

    if len(seen_rirs) != N_SEEN_RIR:

        raise RuntimeError(
            "Jumlah RIR seen tidak sesuai."
        )

    # --------------------------------------------------------
    # UNSEEN
    # --------------------------------------------------------

    unseen_candidates = [
        row["path"]
        for row in rir_table
        if row["identity"]
        in UNSEEN_IDENTITIES
    ]

    if len(unseen_candidates) < N_UNSEEN_RIR:

        raise RuntimeError(
            "Jumlah RIR unseen tidak cukup."
        )

    unseen_rirs = rng.sample(
        unseen_candidates,
        N_UNSEEN_RIR
    )

    return (
        seen_rirs,
        unseen_rirs
    )


# ============================================================
# QC
# ============================================================

def validate_audio(path):

    audio, sr = sf.read(
        str(path),
        always_2d=True
    )

    finite = bool(
        np.isfinite(
            audio
        ).all()
    )

    if not finite:

        return {
            "qc_status": "FAIL",
            "qc_sr": sr,
            "qc_peak": np.nan,
            "qc_rms": np.nan,
            "qc_finite": False
        }

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    audio,
                    dtype=np.float64
                )
            )
        )
    )

    valid = (
        sr == TARGET_SR
        and finite
        and rms > 1e-5
        and peak > 1e-4
        and peak <= 1.0
    )

    return {
        "qc_status":
            "PASS"
            if valid
            else "FAIL",

        "qc_sr":
            sr,

        "qc_peak":
            peak,

        "qc_rms":
            rms,

        "qc_finite":
            finite
    }


# ============================================================
# SOURCE SELECTION
# ============================================================

def select_balanced_sources(
    src,
    split_name,
    n_per_label,
    random_state
):

    subset = src[
        src["split"]
        == split_name
    ].copy()

    counts = (
        subset["label"]
        .value_counts()
    )

    print()
    print(
        f"Available {split_name}:"
    )

    print(
        counts.sort_index()
    )

    for label in [0, 1]:

        available = int(
            counts.get(
                label,
                0
            )
        )

        if available < n_per_label:

            raise RuntimeError(
                f"Split {split_name}, label {label}: "
                f"butuh {n_per_label}, "
                f"tersedia hanya {available}."
            )

    selected = (
        subset
        .groupby(
            "label",
            group_keys=False
        )
        .sample(
            n=n_per_label,
            random_state=random_state
        )
    )

    # Shuffle agar label tidak tersusun
    selected = (
        selected
        .sample(
            frac=1.0,
            random_state=random_state
        )
        .reset_index(
            drop=True
        )
    )

    return selected


# ============================================================
# GENERATE ONE SET
# ============================================================

def generate_set(
    selected,
    output_split,
    rir_group,
    rir_pool,
    start_index,
    rows
):

    total = len(
        selected
    )

    for local_i, (
        _,
        row
    ) in enumerate(
        selected.iterrows()
    ):

        global_i = (
            start_index
            +
            local_i
        )

        source_path = Path(
            row["file_path"]
        )

        source_id = str(
            row[
                "source_file_id"
            ]
        )

        # ----------------------------------------------------
        # LOAD SOURCE
        # ----------------------------------------------------

        wav = load_audio(
            source_path
        )

        # ----------------------------------------------------
        # DETERMINISTIC RIR ASSIGNMENT
        # ----------------------------------------------------

        rir_path = rir_pool[
            local_i
            %
            len(rir_pool)
        ]

        rir_identity = (
            get_rir_identity(
                rir_path
            )
        )

        # ----------------------------------------------------
        # SNR 20 / 10 / 5 BERGANTIAN
        # ----------------------------------------------------

        snr_db = SNR_LIST[
            local_i
            %
            len(SNR_LIST)
        ]

        # ----------------------------------------------------
        # LOAD RIR
        # ----------------------------------------------------

        rir = load_audio(
            rir_path
        )

        # ----------------------------------------------------
        # SIMULATED REPLAY
        # ----------------------------------------------------

        replay = apply_rir(
            wav,
            rir
        )

        replay = add_noise_snr(
            replay,
            snr_db
        )

        replay = normalize_peak(
            replay,
            target_peak=0.95
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        split_dir = (
            OUTPUT_DIR
            /
            output_split
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_name = (
            f"{source_id}"
            f"_sim_"
            f"{rir_group}"
            f"_snr{snr_db}"
            f"_{global_i:05d}.wav"
        )

        output_path = (
            split_dir
            /
            output_name
        )

        sf.write(
            str(output_path),
            replay.squeeze(0)
            .cpu()
            .numpy(),
            TARGET_SR,
            subtype="PCM_16"
        )

        # ----------------------------------------------------
        # QC
        # ----------------------------------------------------

        qc = validate_audio(
            output_path
        )

        # ----------------------------------------------------
        # MANIFEST ROW
        # ----------------------------------------------------

        rows.append({

            "file_id":
                output_path.stem,

            "source_file_id":
                source_id,

            "source_file_path":
                str(source_path),

            "file_path":
                str(output_path),

            "speaker_id":
                row["speaker_id"],

            "attack_id":
                row["attack_id"],

            "label":
                int(row["label"]),

            "dataset":
                row["dataset"],

            "source_split":
                row["split"],

            # normalized experimental split
            "split":
                output_split,

            "condition":
                (
                    "simulated_"
                    + rir_group
                ),

            "rir_file":
                str(rir_path),

            "rir_identity":
                rir_identity,

            "rir_group":
                rir_group,

            "snr_db":
                snr_db,

            "sample_rate":
                TARGET_SR,

            "normalization":
                "peak_0.95",

            "seed":
                SEED,

            **qc
        })

        print(
            f"[{local_i + 1:03d}/{total:03d}] "
            f"{output_split:<10} "
            f"label={int(row['label'])} "
            f"{rir_group:<6} "
            f"{rir_identity:<9} "
            f"SNR={snr_db:2d} "
            f"{qc['qc_status']}"
        )

    return (
        start_index
        +
        total
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=============================================="
    )

    print(
        "WEEK 9 - SIMULATED REPLAY DATASET"
    )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # SOURCE MANIFEST
    # --------------------------------------------------------

    if not SOURCE_MANIFEST.exists():

        raise FileNotFoundError(
            SOURCE_MANIFEST
        )

    src = pd.read_csv(
        SOURCE_MANIFEST
    )

    # Valid binary labels
    src = src[
        src["label"].isin(
            [0, 1]
        )
    ].copy()

    # Normalisasi split
    src["split"] = (
        src["split"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # File harus tersedia
    exists_mask = (
        src["file_path"]
        .apply(
            lambda p:
            Path(str(p)).exists()
        )
    )

    src = src[
        exists_mask
    ].copy()

    print(
        "Source tersedia:",
        len(src)
    )

    # --------------------------------------------------------
    # SELECT BALANCED SOURCE PER SPLIT
    # --------------------------------------------------------

    train_selected = (
        select_balanced_sources(
            src=src,
            split_name="train",
            n_per_label=
                TRAIN_PER_LABEL,
            random_state=
                SEED + 1
        )
    )

    validation_selected = (
        select_balanced_sources(
            src=src,
            split_name="dev",
            n_per_label=
                VALIDATION_PER_LABEL,
            random_state=
                SEED + 2
        )
    )

    test_selected = (
        select_balanced_sources(
            src=src,
            split_name="eval",
            n_per_label=
                TEST_PER_LABEL,
            random_state=
                SEED + 3
        )
    )

    # --------------------------------------------------------
    # SOURCE LEAKAGE CHECK BEFORE GENERATION
    # --------------------------------------------------------

    train_ids = set(
        train_selected[
            "source_file_id"
        ].astype(str)
    )

    val_ids = set(
        validation_selected[
            "source_file_id"
        ].astype(str)
    )

    test_ids = set(
        test_selected[
            "source_file_id"
        ].astype(str)
    )

    if train_ids & val_ids:

        raise RuntimeError(
            "Source leakage TRAIN <-> VALIDATION"
        )

    if train_ids & test_ids:

        raise RuntimeError(
            "Source leakage TRAIN <-> TEST"
        )

    if val_ids & test_ids:

        raise RuntimeError(
            "Source leakage VALIDATION <-> TEST"
        )

    # --------------------------------------------------------
    # RIR
    # --------------------------------------------------------

    rir_table = (
        build_rir_table()
    )

    seen_rirs, unseen_rirs = (
        select_rirs(
            rir_table
        )
    )

    print()
    print(
        "=============================================="
    )

    print(
        "RIR SEEN"
    )

    print(
        "=============================================="
    )

    for rir in seen_rirs:

        print(
            get_rir_identity(
                rir
            ),
            "->",
            rir
        )

    print()
    print(
        "=============================================="
    )

    print(
        "RIR UNSEEN"
    )

    print(
        "=============================================="
    )

    for rir in unseen_rirs:

        print(
            get_rir_identity(
                rir
            ),
            "->",
            rir
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    global_index = 1

    # ========================================================
    # TRAIN
    # Seen RIR only
    # ========================================================

    print()
    print(
        "GENERATING TRAIN SEEN..."
    )

    global_index = generate_set(
        selected=
            train_selected,
        output_split=
            "train",
        rir_group=
            "seen",
        rir_pool=
            seen_rirs,
        start_index=
            global_index,
        rows=
            rows
    )

    # ========================================================
    # VALIDATION
    # Seen RIR only
    # ========================================================

    print()
    print(
        "GENERATING VALIDATION SEEN..."
    )

    global_index = generate_set(
        selected=
            validation_selected,
        output_split=
            "validation",
        rir_group=
            "seen",
        rir_pool=
            seen_rirs,
        start_index=
            global_index,
        rows=
            rows
    )

    # ========================================================
    # TEST
    # UNSEEN RIR only
    # ========================================================

    print()
    print(
        "GENERATING TEST UNSEEN..."
    )

    global_index = generate_set(
        selected=
            test_selected,
        output_split=
            "test",
        rir_group=
            "unseen",
        rir_pool=
            unseen_rirs,
        start_index=
            global_index,
        rows=
            rows
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = pd.DataFrame(
        rows
    )

    manifest.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    # ========================================================
    # FINAL AUDIT
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "WEEK 9 SIMULATED REPLAY SELESAI"
    )

    print(
        "=============================================="
    )

    print(
        "Manifest:",
        OUTPUT_MANIFEST
    )

    print(
        "Total:",
        len(manifest)
    )

    print()

    print(
        "Split:"
    )

    print(
        manifest[
            "split"
        ]
        .value_counts()
    )

    print()

    print(
        "Label:"
    )

    print(
        pd.crosstab(
            manifest[
                "split"
            ],
            manifest[
                "label"
            ]
        )
    )

    print()

    print(
        "RIR group:"
    )

    print(
        pd.crosstab(
            manifest[
                "split"
            ],
            manifest[
                "rir_group"
            ]
        )
    )

    print()

    print(
        "RIR identity:"
    )

    print(
        manifest[
            "rir_identity"
        ]
        .value_counts()
    )

    print()

    print(
        "SNR:"
    )

    print(
        pd.crosstab(
            manifest[
                "split"
            ],
            manifest[
                "snr_db"
            ]
        )
    )

    print()

    print(
        "QC:"
    )

    print(
        manifest[
            "qc_status"
        ]
        .value_counts()
    )

    # --------------------------------------------------------
    # STRICT ASSERTIONS
    # --------------------------------------------------------

    expected_total = (
        TRAIN_PER_LABEL * 2
        +
        VALIDATION_PER_LABEL * 2
        +
        TEST_PER_LABEL * 2
    )

    if len(manifest) != expected_total:

        raise RuntimeError(
            "Jumlah simulated replay "
            "tidak sesuai target."
        )

    if not (
        manifest[
            "qc_status"
        ]
        == "PASS"
    ).all():

        fail_count = int(
            (
                manifest[
                    "qc_status"
                ]
                != "PASS"
            ).sum()
        )

        raise RuntimeError(
            f"Ada {fail_count} "
            "recording gagal QC."
        )

    # --------------------------------------------------------
    # CHECK BALANCE
    # --------------------------------------------------------

    expected_balance = {
        "train": TRAIN_PER_LABEL,
        "validation":
            VALIDATION_PER_LABEL,
        "test": TEST_PER_LABEL
    }

    for split_name, expected in (
        expected_balance.items()
    ):

        part = manifest[
            manifest["split"]
            == split_name
        ]

        counts = (
            part["label"]
            .value_counts()
        )

        if (
            counts.get(0, 0)
            != expected
            or
            counts.get(1, 0)
            != expected
        ):

            raise RuntimeError(
                f"Label imbalance pada "
                f"{split_name}."
            )

    # --------------------------------------------------------
    # RIR LEAKAGE CHECK
    # --------------------------------------------------------

    seen_rir_files = set(
        manifest[
            manifest["rir_group"]
            == "seen"
        ]["rir_file"]
    )

    unseen_rir_files = set(
        manifest[
            manifest["rir_group"]
            == "unseen"
        ]["rir_file"]
    )

    rir_overlap = (
        seen_rir_files
        &
        unseen_rir_files
    )

    if rir_overlap:

        raise RuntimeError(
            "RIR leakage seen/unseen."
        )

    print()
    print(
        "Source leakage : 0"
    )

    print(
        "RIR overlap    : 0"
    )

    print(
        "QC PASS        :",
        (
            manifest["qc_status"]
            == "PASS"
        ).sum()
    )

    print()

    print(
        "HASIL:"
    )

    print(
        "Dataset simulated replay "
        "Week 9 siap digunakan."
    )

if __name__ == "__main__":
    main()