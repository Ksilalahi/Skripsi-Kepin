from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

PLAN_PATH = Path("manifests/week8_real_replay_plan.csv")

EXPECTED_SR = 48000

SILENCE_THRESHOLD = 1e-4
CLIP_THRESHOLD = 0.99

PRE_ROLL_SEC = 0.5
POST_ROLL_SEC = 0.5

DURATION_TOLERANCE_SEC = 0.25


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_object_column(df, column, default=""):
    """
    Memastikan kolom yang akan berisi string / boolean
    menggunakan dtype object agar tidak terjadi error Pandas.
    """

    if column not in df.columns:
        df[column] = default

    df[column] = (
        df[column]
        .fillna("")
        .astype(object)
    )


def ensure_numeric_column(df, column):
    """
    Memastikan kolom numerik tersedia.
    """

    if column not in df.columns:
        df[column] = np.nan

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


def calculate_metrics(audio):
    """
    Menghitung:
    - finite
    - peak
    - RMS
    - clipping ratio
    """

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if audio.size == 0:
        return {
            "finite": False,
            "peak": 0.0,
            "rms": 0.0,
            "clip_ratio": 0.0
        }

    finite = bool(
        np.isfinite(audio).all()
    )

    if not finite:
        return {
            "finite": False,
            "peak": np.nan,
            "rms": np.nan,
            "clip_ratio": np.nan
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

    clip_ratio = float(
        np.mean(
            np.abs(audio) >= CLIP_THRESHOLD
        )
    )

    return {
        "finite": True,
        "peak": peak,
        "rms": rms,
        "clip_ratio": clip_ratio
    }


def get_expected_duration(row):
    """
    Durasi recording yang diharapkan:

    playback
    + 0.5 s pre-roll
    + 0.5 s post-roll
    """

    playback_path = Path(
        str(row["playback_path"])
    )

    if not playback_path.exists():
        raise FileNotFoundError(
            f"Playback tidak ditemukan: {playback_path}"
        )

    info = sf.info(
        playback_path
    )

    playback_duration = (
        info.frames
        / info.samplerate
    )

    expected_duration = (
        playback_duration
        + PRE_ROLL_SEC
        + POST_ROLL_SEC
    )

    return expected_duration


# ============================================================
# QC ONE RECORDING
# ============================================================

def check_recording(row):

    recording_path = Path(
        str(row["recording_path"])
    )

    reasons = []

    # --------------------------------------------------------
    # FILE EXISTENCE
    # --------------------------------------------------------

    if not recording_path.exists():

        return {
            "status": "FAIL",
            "reason": "file_not_found",
            "sr": np.nan,
            "duration": np.nan,
            "peak": np.nan,
            "rms": np.nan,
            "clip_ratio": np.nan,
            "finite": False
        }

    # --------------------------------------------------------
    # READ AUDIO
    # --------------------------------------------------------

    try:

        audio, sr = sf.read(
            recording_path,
            dtype="float32",
            always_2d=False
        )

    except Exception as e:

        return {
            "status": "FAIL",
            "reason": f"read_error:{e}",
            "sr": np.nan,
            "duration": np.nan,
            "peak": np.nan,
            "rms": np.nan,
            "clip_ratio": np.nan,
            "finite": False
        }

    # --------------------------------------------------------
    # CHANNEL
    # --------------------------------------------------------

    if audio.ndim == 2:

        audio = np.mean(
            audio,
            axis=1
        )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        audio
    )

    if sr > 0:
        duration = (
            len(audio)
            / sr
        )
    else:
        duration = np.nan

    # --------------------------------------------------------
    # SAMPLE RATE CHECK
    # --------------------------------------------------------

    if sr != EXPECTED_SR:

        reasons.append(
            f"sr={sr}"
        )

    # --------------------------------------------------------
    # FINITE CHECK
    # --------------------------------------------------------

    if not metrics["finite"]:

        reasons.append(
            "nan_or_inf"
        )

    # --------------------------------------------------------
    # SILENCE CHECK
    # --------------------------------------------------------

    if metrics["finite"]:

        if metrics["rms"] < SILENCE_THRESHOLD:

            reasons.append(
                f"low_rms={metrics['rms']:.8f}"
            )

    # --------------------------------------------------------
    # CLIPPING CHECK
    # --------------------------------------------------------

    if metrics["finite"]:

        if metrics["peak"] >= CLIP_THRESHOLD:

            reasons.append(
                f"peak={metrics['peak']:.6f}"
            )

        if metrics["clip_ratio"] > 0:

            reasons.append(
                f"clip_ratio={metrics['clip_ratio']:.8f}"
            )

    # --------------------------------------------------------
    # DURATION CHECK
    # --------------------------------------------------------

    try:

        expected_duration = (
            get_expected_duration(
                row
            )
        )

        duration_error = abs(
            duration
            - expected_duration
        )

        if duration_error > DURATION_TOLERANCE_SEC:

            reasons.append(
                f"duration={duration:.3f}s "
                f"expected={expected_duration:.3f}s"
            )

    except Exception as e:

        reasons.append(
            f"duration_check_error:{e}"
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if len(reasons) == 0:

        status = "PASS"
        reason = ""

    else:

        status = "FAIL"

        reason = "; ".join(
            reasons
        )

    return {
        "status": status,
        "reason": reason,
        "sr": sr,
        "duration": duration,
        "peak": metrics["peak"],
        "rms": metrics["rms"],
        "clip_ratio": metrics["clip_ratio"],
        "finite": bool(metrics["finite"])
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print("WEEK 8 CONTROLLED REAL REPLAY QC")
    print("==============================================")
    print()

    # --------------------------------------------------------
    # CHECK MANIFEST
    # --------------------------------------------------------

    if not PLAN_PATH.exists():

        raise FileNotFoundError(
            f"Manifest tidak ditemukan: {PLAN_PATH}"
        )

    df = pd.read_csv(
        PLAN_PATH
    )

    print(
        "Manifest:",
        PLAN_PATH
    )

    print(
        "Total jobs:",
        len(df)
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "recording_id",
        "recording_path",
        "playback_path",
        "recording_status"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            f"{missing_columns}"
        )

    # ========================================================
    # PREPARE COLUMN DTYPES
    # ========================================================
    #
    # BAGIAN INI ADALAH PERBAIKAN PENTING.
    #
    # qc_finite dipaksa menjadi object sehingga
    # nilai boolean True / False dapat dimasukkan tanpa
    # TypeError: Invalid value 'True' for dtype 'float64'
    # ========================================================

    ensure_object_column(
        df,
        "qc_status"
    )

    ensure_object_column(
        df,
        "qc_reason"
    )

    ensure_object_column(
        df,
        "qc_finite"
    )

    ensure_numeric_column(
        df,
        "qc_sr"
    )

    ensure_numeric_column(
        df,
        "qc_duration_s"
    )

    ensure_numeric_column(
        df,
        "qc_peak"
    )

    ensure_numeric_column(
        df,
        "qc_rms"
    )

    ensure_numeric_column(
        df,
        "qc_clip_ratio"
    )

    # --------------------------------------------------------
    # FIND DONE RECORDINGS
    # --------------------------------------------------------

    recording_status = (
        df["recording_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    done_indices = df.index[
        recording_status == "done"
    ].tolist()

    print(
        "Recording DONE:",
        len(done_indices)
    )

    print()

    if len(done_indices) == 0:

        print(
            "Belum ada recording dengan status DONE."
        )

        return

    # --------------------------------------------------------
    # RUN QC
    # --------------------------------------------------------

    for number, idx in enumerate(
        done_indices,
        start=1
    ):

        row = df.loc[
            idx
        ]

        recording_id = str(
            row["recording_id"]
        )

        result = check_recording(
            row
        )

        # ----------------------------------------------------
        # SAVE RESULTS TO DATAFRAME
        # ----------------------------------------------------

        df.at[
            idx,
            "qc_status"
        ] = result["status"]

        df.at[
            idx,
            "qc_reason"
        ] = result["reason"]

        df.at[
            idx,
            "qc_sr"
        ] = result["sr"]

        df.at[
            idx,
            "qc_duration_s"
        ] = result["duration"]

        df.at[
            idx,
            "qc_peak"
        ] = result["peak"]

        df.at[
            idx,
            "qc_rms"
        ] = result["rms"]

        df.at[
            idx,
            "qc_clip_ratio"
        ] = result["clip_ratio"]

        # qc_finite sudah dtype object
        df.at[
            idx,
            "qc_finite"
        ] = bool(
            result["finite"]
        )

        # ----------------------------------------------------
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        print(
            f"[{number:03d}/{len(done_indices):03d}] "
            f"{recording_id} "
            f"{result['status']}"
        )

        if np.isfinite(
            result["peak"]
        ):

            print(
                f"   Peak       : "
                f"{result['peak']:.6f}"
            )

        else:

            print(
                "   Peak       : NaN"
            )

        if np.isfinite(
            result["rms"]
        ):

            print(
                f"   RMS        : "
                f"{result['rms']:.6f}"
            )

        else:

            print(
                "   RMS        : NaN"
            )

        if np.isfinite(
            result["clip_ratio"]
        ):

            print(
                f"   Clip ratio : "
                f"{result['clip_ratio']:.8f}"
            )

        else:

            print(
                "   Clip ratio : NaN"
            )

        if np.isfinite(
            result["duration"]
        ):

            print(
                f"   Duration   : "
                f"{result['duration']:.3f} s"
            )

        else:

            print(
                "   Duration   : NaN"
            )

        print(
            "   Finite     :",
            result["finite"]
        )

        if result["reason"]:

            print(
                "   Reason     :",
                result["reason"]
            )

        print()

    # --------------------------------------------------------
    # SAVE UPDATED MANIFEST
    # --------------------------------------------------------

    df.to_csv(
        PLAN_PATH,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    done_df = df.loc[
        done_indices
    ].copy()

    qc_status = (
        done_df["qc_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    pass_count = int(
        (
            qc_status == "PASS"
        ).sum()
    )

    fail_count = int(
        (
            qc_status == "FAIL"
        ).sum()
    )

    clip_values = pd.to_numeric(
        done_df["qc_clip_ratio"],
        errors="coerce"
    )

    clip_count = int(
        (
            clip_values > 0
        ).sum()
    )

    finite_values = (
        done_df["qc_finite"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    finite_false = int(
        (
            finite_values == "false"
        ).sum()
    )

    print(
        "=============================================="
    )

    print(
        "WEEK 8 QC SUMMARY"
    )

    print(
        "=============================================="
    )

    print(
        "DONE checked      :",
        len(done_indices)
    )

    print(
        "PASS              :",
        pass_count
    )

    print(
        "FAIL              :",
        fail_count
    )

    print(
        "Clip ratio > 0    :",
        clip_count
    )

    print(
        "Finite False      :",
        finite_false
    )

    print()

    print(
        "Manifest updated  :",
        PLAN_PATH
    )

    print()

    if fail_count == 0:

        print(
            "Semua recording DONE yang diperiksa "
            "lolos QC."
        )

    else:

        print(
            "Ada recording yang FAIL."
        )

        print(
            "Periksa kolom qc_reason pada manifest "
            "dan rekam ulang recording tersebut."
        )

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()