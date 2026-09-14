from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


PLAN_PATH = Path("manifests/week7_real_replay_plan.csv")
EXPECTED_SR = 48000
SILENCE_THRESHOLD = 1e-4
CLIP_THRESHOLD = 0.99
DURATION_TOLERANCE = 0.25

def check_audio(
    wav_path,
    expected_duration=None
):

    wav_path = Path(wav_path)
    result = {
        "qc_status": "FAIL",
        "qc_reason": "",
        "qc_sr": np.nan,
        "qc_duration_s": np.nan,
        "qc_peak": np.nan,
        "qc_rms": np.nan,
        "qc_clip_ratio": np.nan,
        "qc_finite": False
    }

    if not wav_path.exists():

        result["qc_reason"] = (
            "file_not_found"
        )
        return result

    try:
        audio, sr = sf.read(
            wav_path,
            always_2d=True,
            dtype="float32"
        )
    except Exception as e:
        result["qc_reason"] = (
            f"read_error: {e}"
        )
        return result

    if len(audio) == 0:
        result["qc_reason"] = (
            "empty_audio"
        )
        return result

    finite = bool(
        np.isfinite(
            audio
        ).all()
    )

    result["qc_finite"] = finite
    if not finite:
        result["qc_reason"] = (
            "nan_or_inf"
        )
        return result

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
        )
    )

    clip_ratio = float(
        np.mean(
            np.abs(
                audio
            )
            >=
            CLIP_THRESHOLD
        )
    )

    duration_s = float(
        len(audio)
        /
        sr
    )

    result["qc_sr"] = sr
    result["qc_duration_s"] = duration_s
    result["qc_peak"] = peak
    result["qc_rms"] = rms
    result["qc_clip_ratio"] = clip_ratio

    if sr != EXPECTED_SR:
        result["qc_reason"] = (
            f"invalid_sr_{sr}"
        )
        return result

    if rms < SILENCE_THRESHOLD:
        result["qc_reason"] = (
            f"silence_rms_{rms:.8f}"
        )
        return result

    if clip_ratio > 0:
        result["qc_reason"] = (
            f"clipping_peak_{peak:.6f}"
            f"_ratio_{clip_ratio:.8f}"
        )
        return result

    if (
        expected_duration
        is not None
        and
        np.isfinite(
            expected_duration
        )
    ):

        diff = abs(
            duration_s
            -
            expected_duration
        )

        if (
            diff
            >
            DURATION_TOLERANCE
        ):

            result["qc_reason"] = (
                "duration_mismatch_"
                f"{duration_s:.3f}"
            )

            return result

    result["qc_status"] = "PASS"
    result["qc_reason"] = ""
    return result

def main():
    if not PLAN_PATH.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan: "
            f"{PLAN_PATH}"
        )

    df = pd.read_csv(PLAN_PATH)
    text_columns = [
        "qc_status",
        "qc_reason"
    ]

    for col in text_columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = (
            df[col]
            .fillna("")
            .astype(object)
        )

    numeric_columns = [
        "qc_sr",
        "qc_duration_s",
        "qc_peak",
        "qc_rms",
        "qc_clip_ratio"
    ]

    for col in numeric_columns:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    if (
        "qc_finite"
        not in df.columns
    ):

        df[
            "qc_finite"
        ] = False

    print()
    print("=======================================================")
    print("WEEK 7 REAL REPLAY QUALITY CONTROL")
    print("=======================================================")

    checked = 0
    passed = 0
    failed = 0

    for idx, row in df.iterrows():
        recording_status = str(
            row[
                "recording_status"
            ]
        ).strip().lower()

        if (
            recording_status
            !=
            "done"
        ):

            continue

        recording_id = str(
            row[
                "recording_id"
            ]
        )

        wav_path = Path(
            str(
                row[
                    "output_path"
                ]
            )
        )

        expected_duration = None
        if (
            "duration_s"
            in df.columns
        ):
            try:
                source_duration = float(
                    row[
                        "duration_s"
                    ]
                )

                expected_duration = (
                    source_duration
                    +
                    1.0
                )
            except Exception:
                expected_duration = None

        result = check_audio(
            wav_path,
            expected_duration
        )

        df.at[
            idx,
            "qc_status"
        ] = result[
            "qc_status"
        ]

        df.at[
            idx,
            "qc_reason"
        ] = result[
            "qc_reason"
        ]

        df.at[
            idx,
            "qc_sr"
        ] = result[
            "qc_sr"
        ]

        df.at[
            idx,
            "qc_duration_s"
        ] = result[
            "qc_duration_s"
        ]

        df.at[
            idx,
            "qc_peak"
        ] = result[
            "qc_peak"
        ]

        df.at[
            idx,
            "qc_rms"
        ] = result[
            "qc_rms"
        ]

        df.at[
            idx,
            "qc_clip_ratio"
        ] = result[
            "qc_clip_ratio"
        ]

        df.at[
            idx,
            "qc_finite"
        ] = result[
            "qc_finite"
        ]

        checked += 1
        if (
            result[
                "qc_status"
            ]
            ==
            "PASS"
        ):

            passed += 1
            print(
                f"{recording_id}: PASS"
                f" | sr={int(result['qc_sr'])}"
                f" | dur={result['qc_duration_s']:.3f}"
                f" | peak={result['qc_peak']:.6f}"
                f" | rms={result['qc_rms']:.6f}"
                f" | clip={result['qc_clip_ratio']:.8f}"
            )

        else:
            failed += 1
            print(
                f"{recording_id}: FAIL"
                f" | reason="
                f"{result['qc_reason']}"
            )

    df.to_csv(
        PLAN_PATH,
        index=False
    )
    print()
    print("=======================================================")
    print("QC SUMMARY")
    print("=======================================================")
    print("Checked :",checked)
    print("PASS    :",passed)
    print("FAIL    :",failed)
    print("Manifest:",PLAN_PATH)
    print("=======================================================")

if __name__ == "__main__":
    main()