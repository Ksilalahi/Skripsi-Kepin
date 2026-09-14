from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


PLAN_PATH = Path(
    "manifests/pilot_acquisition_plan.csv"
)

OUTPUT_QC = Path(
    "manifests/pilot_real_replay_qc.csv"
)


EXPECTED_SR = 48000

SILENCE_RMS_THRESHOLD = 1e-4

CLIP_THRESHOLD = 0.99


def qc_audio(path):

    result = {
        "qc_exists": False,
        "qc_readable": False,
        "qc_sr": None,
        "qc_duration_s": None,
        "qc_peak": None,
        "qc_rms": None,
        "qc_clip_ratio": None,
        "qc_finite": False,
        "qc_silent": True,
        "qc_clipping": False,
        "qc_status": "FAIL",
        "qc_reason": ""
    }

    path = Path(path)

    # ========================================================
    # FILE EXISTS
    # ========================================================

    if not path.exists():

        result[
            "qc_reason"
        ] = "file_not_found"

        return result


    result[
        "qc_exists"
    ] = True


    # ========================================================
    # READ AUDIO
    # ========================================================

    try:

        audio, sr = sf.read(
            path,
            always_2d=True,
            dtype="float32"
        )

    except Exception as e:

        result[
            "qc_reason"
        ] = (
            f"read_error:{e}"
        )

        return result


    result[
        "qc_readable"
    ] = True


    # ========================================================
    # BASIC INFO
    # ========================================================

    duration = (
        len(audio) / sr
    )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    rms = float(
        np.sqrt(
            np.mean(
                audio ** 2
            )
        )
    )


    finite = bool(
        np.isfinite(
            audio
        ).all()
    )


    clip_ratio = float(
        np.mean(
            np.abs(audio)
            >= CLIP_THRESHOLD
        )
    )


    silent = (
        rms
        < SILENCE_RMS_THRESHOLD
    )


    clipping = (
        clip_ratio > 0
    )


    # ========================================================
    # SAVE QC VALUES
    # ========================================================

    result[
        "qc_sr"
    ] = int(sr)

    result[
        "qc_duration_s"
    ] = float(duration)

    result[
        "qc_peak"
    ] = peak

    result[
        "qc_rms"
    ] = rms

    result[
        "qc_clip_ratio"
    ] = clip_ratio

    result[
        "qc_finite"
    ] = finite

    result[
        "qc_silent"
    ] = bool(silent)

    result[
        "qc_clipping"
    ] = bool(clipping)


    # ========================================================
    # DETERMINE FAIL REASONS
    # ========================================================

    reasons = []


    if sr != EXPECTED_SR:

        reasons.append(
            "wrong_sample_rate"
        )


    if duration <= 0:

        reasons.append(
            "invalid_duration"
        )


    if not finite:

        reasons.append(
            "nan_or_inf"
        )


    if silent:

        reasons.append(
            "silence"
        )


    if clipping:

        reasons.append(
            "clipping"
        )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    if len(reasons) == 0:

        result[
            "qc_status"
        ] = "PASS"

        result[
            "qc_reason"
        ] = "ok"

    else:

        result[
            "qc_status"
        ] = "FAIL"

        result[
            "qc_reason"
        ] = ";".join(
            reasons
        )


    return result


def main():

    if not PLAN_PATH.exists():

        raise FileNotFoundError(
            f"Tidak ditemukan: "
            f"{PLAN_PATH}"
        )


    df = pd.read_csv(
        PLAN_PATH
    )


    required = [
        "pilot_id",
        "output_path",
        "recording_status"
    ]


    missing = [
        col
        for col in required
        if col not in df.columns
    ]


    if missing:

        raise ValueError(
            "Kolom tidak ditemukan: "
            + ", ".join(
                missing
            )
        )


    rows = []


    print(
        "===================================="
    )

    print(
        "REAL REPLAY PILOT QC"
    )

    print(
        "===================================="
    )


    for _, row in df.iterrows():

        pilot_id = str(
            row[
                "pilot_id"
            ]
        )

        output_path = str(
            row[
                "output_path"
            ]
        )


        qc = qc_audio(
            output_path
        )


        result = (
            row.to_dict()
        )

        result.update(
            qc
        )

        rows.append(
            result
        )


        print(
            f"{pilot_id}: "
            f"{qc['qc_status']} "
            f"| sr={qc['qc_sr']} "
            f"| duration="
            f"{qc['qc_duration_s']} "
            f"| peak="
            f"{qc['qc_peak']} "
            f"| rms="
            f"{qc['qc_rms']} "
            f"| clip="
            f"{qc['qc_clip_ratio']} "
            f"| reason="
            f"{qc['qc_reason']}"
        )


    qc_df = pd.DataFrame(
        rows
    )


    OUTPUT_QC.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    qc_df.to_csv(
        OUTPUT_QC,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "===================================="
    )

    print(
        "QC SUMMARY"
    )

    print(
        "===================================="
    )


    print(
        qc_df[
            "qc_status"
        ].value_counts()
    )


    print()
    print(
        "FAIL FILES:"
    )


    fail_df = qc_df[
        qc_df[
            "qc_status"
        ]
        ==
        "FAIL"
    ]


    if len(
        fail_df
    ) == 0:

        print(
            "NONE"
        )

    else:

        print(
            fail_df[
                [
                    "pilot_id",
                    "qc_reason",
                    "qc_peak",
                    "qc_rms",
                    "qc_clip_ratio"
                ]
            ].to_string(
                index=False
            )
        )


    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_QC
    )


if __name__ == "__main__":

    main()