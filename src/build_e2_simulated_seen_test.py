from pathlib import Path
import hashlib
import random

import numpy as np
import pandas as pd
import soundfile as sf

from scipy.signal import fftconvolve, resample_poly


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SR = 16000

SOURCE_MANIFEST = Path(
    "manifests/source_manifest.csv"
)

E2_TRAIN_MANIFEST = Path(
    "manifests/train_e2_clean_manifest.csv"
)

WEEK9_SIM_MANIFEST = Path(
    "manifests/manifest_replay_sim_week9.csv"
)

RIR_ROOT = Path(
    "data/rir/original"
)

OUTPUT_DIR = Path(
    "data/processed/replay_sim_e2_seen_test"
)

OUTPUT_MANIFEST = Path(
    "manifests/e2_simulated_seen_test.csv"
)

SEED = 2026

N_PER_LABEL = 100

SEEN_IDENTITIES = [
    "booth",
    "lecture",
    "meeting",
    "office",
    "phone"
]

SNR_LEVELS = [
    20,
    10,
    5
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    SEED
)

np.random.seed(
    SEED
)


# ============================================================
# SHA256
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


# ============================================================
# NORMALIZE SPLIT
# ============================================================

def normalize_split(value):

    value = str(
        value
    ).strip().lower()

    if value == "train":
        return "train"

    if value in [
        "dev",
        "validation",
        "val"
    ]:
        return "validation"

    if value in [
        "eval",
        "test"
    ]:
        return "test"

    return value


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(
    path,
    target_sr=TARGET_SR
):

    audio, sr = sf.read(
        str(path),
        always_2d=True
    )

    # --------------------------------------------
    # Multichannel -> mono
    # --------------------------------------------

    audio = audio.mean(
        axis=1
    )

    audio = audio.astype(
        np.float32
    )

    # --------------------------------------------
    # Resample
    # --------------------------------------------

    if sr != target_sr:

        audio = resample_poly(
            audio,
            target_sr,
            sr
        )

        audio = audio.astype(
            np.float32
        )

    return (
        audio,
        target_sr
    )


# ============================================================
# LOAD RIR
# ============================================================

def load_rir(
    path,
    target_sr=TARGET_SR
):

    rir, sr = sf.read(
        str(path),
        always_2d=True
    )

    rir = rir.mean(
        axis=1
    )

    rir = rir.astype(
        np.float32
    )

    if sr != target_sr:

        rir = resample_poly(
            rir,
            target_sr,
            sr
        )

        rir = rir.astype(
            np.float32
        )

    # --------------------------------------------
    # Remove pathological all-zero RIR
    # --------------------------------------------

    peak = float(
        np.max(
            np.abs(
                rir
            )
        )
    )

    if peak <= 1e-8:

        raise RuntimeError(
            f"RIR hampir silent: {path}"
        )

    # --------------------------------------------
    # Normalize RIR conservatively
    # --------------------------------------------

    rir = (
        rir
        /
        peak
    )

    return rir


# ============================================================
# APPLY RIR
# ============================================================

def apply_rir(
    audio,
    rir
):

    replay = fftconvolve(
        audio,
        rir,
        mode="full"
    )

    # Potong ke durasi source semula.
    replay = replay[
        :len(audio)
    ]

    replay = replay.astype(
        np.float32
    )

    return replay


# ============================================================
# ADD WHITE NOISE AT TARGET SNR
# ============================================================

def add_noise_snr(
    audio,
    snr_db,
    rng
):

    signal_power = float(
        np.mean(
            audio ** 2
        )
    )

    if signal_power <= 1e-12:

        raise RuntimeError(
            "Signal power terlalu kecil."
        )

    noise = rng.normal(
        0.0,
        1.0,
        size=audio.shape
    ).astype(
        np.float32
    )

    noise_power = float(
        np.mean(
            noise ** 2
        )
    )

    target_noise_power = (
        signal_power
        /
        (
            10.0
            **
            (
                snr_db
                /
                10.0
            )
        )
    )

    scale = np.sqrt(
        target_noise_power
        /
        (
            noise_power
            +
            1e-12
        )
    )

    noisy = (
        audio
        +
        noise
        *
        scale
    )

    return noisy.astype(
        np.float32
    )


# ============================================================
# CONSERVATIVE PEAK NORMALIZATION
# ============================================================

def normalize_peak(
    audio,
    target_peak=0.95
):

    peak = float(
        np.max(
            np.abs(
                audio
            )
        )
    )

    if peak <= 1e-8:

        return audio.astype(
            np.float32
        )

    # Hanya turunkan level jika terlalu tinggi.
    if peak > target_peak:

        audio = (
            audio
            *
            (
                target_peak
                /
                peak
            )
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# QC
# ============================================================

def qc_audio(
    audio
):

    finite = bool(
        np.isfinite(
            audio
        ).all()
    )

    peak = float(
        np.max(
            np.abs(
                audio
            )
        )
    )

    rms = float(
        np.sqrt(
            np.mean(
                audio ** 2
            )
            +
            1e-12
        )
    )

    clip_ratio = float(
        np.mean(
            np.abs(
                audio
            )
            >=
            0.999
        )
    )

    if not finite:

        return (
            "FAIL",
            "non_finite",
            peak,
            rms,
            clip_ratio
        )

    if rms < 1e-5:

        return (
            "FAIL",
            "too_silent",
            peak,
            rms,
            clip_ratio
        )

    if clip_ratio > 0.001:

        return (
            "FAIL",
            "clipping",
            peak,
            rms,
            clip_ratio
        )

    return (
        "PASS",
        "",
        peak,
        rms,
        clip_ratio
    )


# ============================================================
# FIND EXACT SEEN RIR SET
# ============================================================

def select_seen_rirs():

    if not RIR_ROOT.exists():

        raise FileNotFoundError(
            RIR_ROOT
        )

    rir_files = sorted(
        RIR_ROOT.rglob(
            "*.wav"
        )
    )

    if len(
        rir_files
    ) == 0:

        raise RuntimeError(
            "Tidak ada RIR WAV ditemukan."
        )

    selected = {}

    # --------------------------------------------------------
    # Gunakan rule deterministik yang sama:
    # satu RIR pertama hasil sorting untuk setiap identity.
    #
    # Dengan struktur AIR yang kita gunakan sebelumnya,
    # ini menjaga set 5 identity seen tetap:
    # booth, lecture, meeting, office, phone.
    # --------------------------------------------------------

    for identity in SEEN_IDENTITIES:

        candidates = [

            path

            for path in rir_files

            if identity.lower()
            in
            path.name.lower()
        ]

        if len(
            candidates
        ) == 0:

            raise RuntimeError(
                f"RIR identity tidak ditemukan: "
                f"{identity}"
            )

        selected[
            identity
        ] = candidates[
            0
        ]

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=============================================="
    )

    print(
        "E2 - BUILD SOURCE-DISJOINT SIMULATED SEEN TEST"
    )

    print(
        "=============================================="
    )

    # ========================================================
    # CHECK INPUTS
    # ========================================================

    required_files = [
        SOURCE_MANIFEST,
        E2_TRAIN_MANIFEST
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # LOAD MANIFESTS
    # ========================================================

    source = pd.read_csv(
        SOURCE_MANIFEST
    )

    e2 = pd.read_csv(
        E2_TRAIN_MANIFEST
    )

    print()
    print(
        "Source manifest rows:",
        len(
            source
        )
    )

    print(
        "E2 train+validation rows:",
        len(
            e2
        )
    )

    # ========================================================
    # VALIDATE SOURCE MANIFEST
    # ========================================================

    source_required = [
        "source_file_id",
        "file_path",
        "speaker_id",
        "attack_id",
        "label",
        "dataset",
        "split"
    ]

    missing = [

        column

        for column in source_required

        if column not in source.columns
    ]

    if missing:

        raise ValueError(
            "source_manifest.csv kehilangan "
            f"kolom: {missing}"
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    source[
        "label"
    ] = pd.to_numeric(
        source[
            "label"
        ],
        errors="raise"
    ).astype(int)

    source[
        "normalized_split"
    ] = source[
        "split"
    ].apply(
        normalize_split
    )

    # ========================================================
    # EXCLUDE ALL E2 TRAIN/VALIDATION SOURCES
    # ========================================================

    used_e2_sources = set(
        e2[
            "source_file_id"
        ]
        .astype(str)
    )

    print(
        "E2 used source:",
        len(
            used_e2_sources
        )
    )

    # ========================================================
    # TEST CANDIDATES
    #
    # Hanya source dari eval/test source split.
    # Tidak boleh overlap E2 train/validation.
    # ========================================================

    candidates = source[
        source[
            "normalized_split"
        ]
        ==
        "test"
    ].copy()

    candidates = candidates[
        ~candidates[
            "source_file_id"
        ]
        .astype(str)
        .isin(
            used_e2_sources
        )
    ].copy()

    # ========================================================
    # VALID FILES ONLY
    # ========================================================

    candidates[
        "_exists"
    ] = candidates[
        "file_path"
    ].apply(
        lambda x:
            Path(
                str(x)
            ).exists()
    )

    missing_source_audio = int(
        (
            ~candidates[
                "_exists"
            ]
        ).sum()
    )

    print()
    print(
        "Candidate eval/test:",
        len(
            candidates
        )
    )

    print(
        "Missing candidate audio:",
        missing_source_audio
    )

    candidates = candidates[
        candidates[
            "_exists"
        ]
    ].copy()

    # ========================================================
    # REMOVE DUPLICATE SOURCE IDs
    # ========================================================

    candidates = (
        candidates
        .drop_duplicates(
            subset=[
                "source_file_id"
            ],
            keep="first"
        )
        .copy()
    )

    # ========================================================
    # AUDIT LABEL AVAILABILITY
    # ========================================================

    print()
    print(
        "Candidate label distribution:"
    )

    print(
        candidates[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # SAMPLE 100 + 100
    # ========================================================

    selected_parts = []

    for label in [
        0,
        1
    ]:

        label_df = candidates[
            candidates[
                "label"
            ]
            ==
            label
        ].copy()

        if len(
            label_df
        ) < N_PER_LABEL:

            raise RuntimeError(
                f"Candidate label {label} hanya "
                f"{len(label_df)}, "
                f"butuh {N_PER_LABEL}."
            )

        sampled = label_df.sample(
            n=N_PER_LABEL,
            random_state=(
                SEED
                +
                label
            )
        )

        selected_parts.append(
            sampled
        )

    selected_sources = pd.concat(
        selected_parts,
        ignore_index=True
    )

    # Shuffle deterministically
    selected_sources = (
        selected_sources
        .sample(
            frac=1.0,
            random_state=SEED
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "Selected sources:",
        len(
            selected_sources
        )
    )

    print(
        "Selected labels:"
    )

    print(
        selected_sources[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # SOURCE LEAKAGE CHECK
    # ========================================================

    selected_ids = set(
        selected_sources[
            "source_file_id"
        ]
        .astype(str)
    )

    overlap = (
        selected_ids
        &
        used_e2_sources
    )

    print()
    print(
        "Source overlap dengan E2 train/validation:",
        len(
            overlap
        )
    )

    if overlap:

        raise RuntimeError(
            "Source leakage ditemukan."
        )

    # ========================================================
    # SELECT 5 SEEN RIRS
    # ========================================================

    seen_rirs = select_seen_rirs()

    print()
    print(
        "Seen RIR set:"
    )

    for identity in SEEN_IDENTITIES:

        print(
            f"{identity:8s} -> "
            f"{seen_rirs[identity]}"
        )

    if len(
        seen_rirs
    ) != 5:

        raise RuntimeError(
            "Seen RIR harus 5."
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # GENERATE 200 SIMULATED REPLAY
    # ========================================================

    result_rows = []

    total = len(
        selected_sources
    )

    print()
    print(
        "=============================================="
    )

    print(
        "GENERATING E2 SIMULATED SEEN TEST"
    )

    print(
        "=============================================="
    )

    for index, row in (
        selected_sources.iterrows()
    ):

        source_id = str(
            row[
                "source_file_id"
            ]
        )

        # ----------------------------------------------------
        # Balanced deterministic RIR cycling
        #
        # 200 / 5 = 40 samples per RIR identity.
        # ----------------------------------------------------

        rir_identity = (
            SEEN_IDENTITIES[
                index
                %
                len(
                    SEEN_IDENTITIES
                )
            ]
        )

        rir_path = (
            seen_rirs[
                rir_identity
            ]
        )

        # ----------------------------------------------------
        # Balanced SNR cycling
        #
        # 67 / 67 / 66 approximately.
        # ----------------------------------------------------

        snr_db = (
            SNR_LEVELS[
                index
                %
                len(
                    SNR_LEVELS
                )
            ]
        )

        sample_seed = (
            SEED
            +
            index
        )

        rng = np.random.default_rng(
            sample_seed
        )

        # ----------------------------------------------------
        # LOAD SOURCE
        # ----------------------------------------------------

        audio, sr = load_audio(
            row[
                "file_path"
            ]
        )

        # ----------------------------------------------------
        # LOAD RIR
        # ----------------------------------------------------

        rir = load_rir(
            rir_path
        )

        # ----------------------------------------------------
        # CONVOLUTION
        # ----------------------------------------------------

        replay = apply_rir(
            audio,
            rir
        )

        # ----------------------------------------------------
        # ADD NOISE
        # ----------------------------------------------------

        replay = add_noise_snr(
            replay,
            snr_db,
            rng
        )

        # ----------------------------------------------------
        # CONSERVATIVE NORMALIZATION
        # ----------------------------------------------------

        replay = normalize_peak(
            replay,
            target_peak=0.95
        )

        # ----------------------------------------------------
        # QC
        # ----------------------------------------------------

        (
            qc_status,
            qc_reason,
            qc_peak,
            qc_rms,
            qc_clip_ratio
        ) = qc_audio(
            replay
        )

        # ----------------------------------------------------
        # OUTPUT ID/PATH
        # ----------------------------------------------------

        file_id = (
            f"E2_SIM_SEEN_"
            f"{index:04d}_"
            f"{source_id}_"
            f"{rir_identity}_"
            f"snr{snr_db}"
        )

        output_path = (
            OUTPUT_DIR
            /
            f"{file_id}.wav"
        )

        # ----------------------------------------------------
        # WRITE 16 kHz / PCM16
        # ----------------------------------------------------

        sf.write(
            str(
                output_path
            ),
            replay,
            TARGET_SR,
            subtype="PCM_16"
        )

        # ----------------------------------------------------
        # SHA
        # ----------------------------------------------------

        file_sha256 = sha256_file(
            output_path
        )

        duration_s = (
            len(
                replay
            )
            /
            TARGET_SR
        )

        result_rows.append({

            "file_id":
                file_id,

            "source_file_id":
                source_id,

            "file_path":
                str(
                    output_path
                ),

            "source_file_path":
                str(
                    row[
                        "file_path"
                    ]
                ),

            "speaker_id":
                row[
                    "speaker_id"
                ],

            "attack_id":
                row[
                    "attack_id"
                ],

            "label":
                int(
                    row[
                        "label"
                    ]
                ),

            "dataset":
                row[
                    "dataset"
                ],

            "source_split":
                row[
                    "split"
                ],

            "split":
                "test",

            "condition":
                "simulated_replay",

            "rir_identity":
                rir_identity,

            "rir_group":
                "seen",

            "rir_path":
                str(
                    rir_path
                ),

            "snr_db":
                int(
                    snr_db
                ),

            "seed":
                int(
                    sample_seed
                ),

            "sample_rate":
                TARGET_SR,

            "duration_s":
                duration_s,

            "sha256":
                file_sha256,

            "qc_status":
                qc_status,

            "qc_reason":
                qc_reason,

            "qc_peak":
                qc_peak,

            "qc_rms":
                qc_rms,

            "qc_clip_ratio":
                qc_clip_ratio
        })

        if (
            (index + 1) % 20 == 0
            or
            (index + 1) == total
        ):

            print(
                f"[{index + 1}/{total}]"
            )

    # ========================================================
    # BUILD MANIFEST
    # ========================================================

    result = pd.DataFrame(
        result_rows
    )

    # ========================================================
    # FINAL QC AUDIT
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "E2 SIMULATED SEEN TEST AUDIT"
    )

    print(
        "=============================================="
    )

    print(
        "Rows:",
        len(
            result
        )
    )

    print(
        "Unique source:",
        result[
            "source_file_id"
        ].nunique()
    )

    print(
        "Unique file:",
        result[
            "file_id"
        ].nunique()
    )

    print()
    print(
        "Label:"
    )

    print(
        result[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "RIR identity:"
    )

    print(
        result[
            "rir_identity"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "SNR:"
    )

    print(
        result[
            "snr_db"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "QC:"
    )

    print(
        result[
            "qc_status"
        ]
        .value_counts()
    )

    # ========================================================
    # STRICT ASSERTIONS
    # ========================================================

    if len(
        result
    ) != 200:

        raise RuntimeError(
            "Output harus 200."
        )

    if (
        result[
            "source_file_id"
        ]
        .nunique()
        !=
        200
    ):

        raise RuntimeError(
            "Harus 200 unique source."
        )

    if (
        result[
            "file_id"
        ]
        .nunique()
        !=
        200
    ):

        raise RuntimeError(
            "Harus 200 unique file."
        )

    if int(
        (
            result[
                "label"
            ]
            ==
            0
        ).sum()
    ) != 100:

        raise RuntimeError(
            "Label 0 harus 100."
        )

    if int(
        (
            result[
                "label"
            ]
            ==
            1
        ).sum()
    ) != 100:

        raise RuntimeError(
            "Label 1 harus 100."
        )

    # --------------------------------------------------------
    # 200 / 5 = exactly 40 per RIR
    # --------------------------------------------------------

    rir_counts = (
        result[
            "rir_identity"
        ]
        .value_counts()
    )

    for identity in SEEN_IDENTITIES:

        actual = int(
            rir_counts.get(
                identity,
                0
            )
        )

        if actual != 40:

            raise RuntimeError(
                f"RIR {identity} harus 40, "
                f"ditemukan {actual}."
            )

    if not (
        result[
            "qc_status"
        ]
        ==
        "PASS"
    ).all():

        failed = result[
            result[
                "qc_status"
            ]
            !=
            "PASS"
        ]

        print()
        print(
            "FAILED QC:"
        )

        print(
            failed[
                [
                    "file_id",
                    "qc_reason"
                ]
            ]
        )

        raise RuntimeError(
            "Ada E2 simulated replay "
            "yang gagal QC."
        )

    # ========================================================
    # FINAL LEAKAGE CHECK AGAIN
    # ========================================================

    final_ids = set(
        result[
            "source_file_id"
        ]
        .astype(str)
    )

    final_overlap = (
        final_ids
        &
        used_e2_sources
    )

    if final_overlap:

        raise RuntimeError(
            "Final E2 test leakage ditemukan."
        )

    # ========================================================
    # SAVE
    # ========================================================

    result.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "E2 SIMULATED SEEN TEST BERHASIL DIBUAT"
    )

    print(
        "=============================================="
    )

    print(
        "Output manifest:",
        OUTPUT_MANIFEST
    )

    print(
        "Output audio:",
        OUTPUT_DIR
    )

    print(
        "Rows:",
        len(
            result
        )
    )

    print(
        "Source overlap dengan E2 train/validation:",
        len(
            final_overlap
        )
    )

    print(
        "QC PASS:",
        int(
            (
                result[
                    "qc_status"
                ]
                ==
                "PASS"
            ).sum()
        )
    )

if __name__ == "__main__":

    main()