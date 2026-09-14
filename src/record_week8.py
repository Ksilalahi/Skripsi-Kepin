from pathlib import Path
import time
import threading

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

PLAN_PATH = Path("manifests/week8_real_replay_plan.csv")

RECORD_SR = 48000
RECORD_SUBTYPE = "PCM_24"

# Device yang sebelumnya berhasil digunakan
INPUT_DEVICE = 1
OUTPUT_DEVICE = 3

PRE_ROLL_SEC = 0.5
POST_ROLL_SEC = 0.5

BLOCKSIZE = 1024

# QC sederhana saat recording
SILENCE_THRESHOLD = 1e-4
CLIP_THRESHOLD = 0.99

# Berapa recording baru per satu kali menjalankan script
MAX_NEW_RECORDINGS = 30

# Sedikit waktu agar input stream sudah aktif sebelum playback
PLAYBACK_DELAY_SEC = 0.10


# ============================================================
# HELPER
# ============================================================

def ensure_object_column(df, column, default=""):
    """
    Memastikan kolom text tidak bertipe float karena nilai kosong/NaN.
    """
    if column not in df.columns:
        df[column] = default

    df[column] = df[column].fillna("").astype(object)


def ensure_numeric_column(df, column):
    """
    Membuat kolom numerik jika belum tersedia.
    """
    if column not in df.columns:
        df[column] = np.nan


def load_playback_audio(path):
    """
    Membaca playback master.

    Playback master Week 7/8 seharusnya:
    - 48 kHz
    - mono
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Playback file tidak ditemukan: {path}"
        )

    audio, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False
    )

    if sr != RECORD_SR:
        raise ValueError(
            f"Sample rate playback {sr} Hz, "
            f"seharusnya {RECORD_SR} Hz."
        )

    # Jika stereo, ubah ke mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    if not np.isfinite(audio).all():
        raise ValueError(
            "Playback mengandung NaN atau Inf."
        )

    if len(audio) == 0:
        raise ValueError(
            "Playback kosong."
        )

    return audio


def calculate_audio_metrics(audio):
    """
    Menghitung statistik recording:
    peak, RMS, clipping ratio, finite.
    """

    audio = np.asarray(audio, dtype=np.float32)

    finite = bool(np.isfinite(audio).all())

    if len(audio) == 0:
        return {
            "peak": 0.0,
            "rms": 0.0,
            "clip_ratio": 0.0,
            "finite": False
        }

    peak = float(
        np.max(np.abs(audio))
    )

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio, dtype=np.float64)
            )
        )
    )

    clip_ratio = float(
        np.mean(
            np.abs(audio) >= CLIP_THRESHOLD
        )
    )

    return {
        "peak": peak,
        "rms": rms,
        "clip_ratio": clip_ratio,
        "finite": finite
    }


# ============================================================
# RECORDING
# ============================================================

def play_and_record(playback):
    """
    Melakukan playback dan recording menggunakan
    InputStream + OutputStream secara eksplisit.

    Metode ini digunakan karena sd.playrec sebelumnya
    menghasilkan silence pada perangkat user.
    """

    pre_samples = int(
        PRE_ROLL_SEC * RECORD_SR
    )

    post_samples = int(
        POST_ROLL_SEC * RECORD_SR
    )

    total_samples = (
        pre_samples
        + len(playback)
        + post_samples
    )

    recorded_blocks = []

    playback_done = threading.Event()

    # --------------------------------------------------------
    # INPUT / RECORDING
    # --------------------------------------------------------

    def input_callback(
        indata,
        frames,
        time_info,
        status
    ):
        if status:
            print(
                "[InputStream status]",
                status
            )

        recorded_blocks.append(
            indata.copy()
        )

    # --------------------------------------------------------
    # OUTPUT / PLAYBACK
    # --------------------------------------------------------

    playback_position = 0

    # tambahkan pre-roll silence
    playback_buffer = np.concatenate(
        [
            np.zeros(
                pre_samples,
                dtype=np.float32
            ),
            playback,
            np.zeros(
                post_samples,
                dtype=np.float32
            )
        ]
    )

    def output_callback(
        outdata,
        frames,
        time_info,
        status
    ):
        nonlocal playback_position

        if status:
            print(
                "[OutputStream status]",
                status
            )

        remaining = (
            len(playback_buffer)
            - playback_position
        )

        count = min(
            frames,
            remaining
        )

        outdata.fill(0)

        if count > 0:
            outdata[:count, 0] = (
                playback_buffer[
                    playback_position:
                    playback_position + count
                ]
            )

            playback_position += count

        if playback_position >= len(playback_buffer):
            playback_done.set()
            raise sd.CallbackStop


    # --------------------------------------------------------
    # OPEN STREAMS
    # --------------------------------------------------------

    with sd.InputStream(
        samplerate=RECORD_SR,
        device=INPUT_DEVICE,
        channels=1,
        dtype="float32",
        blocksize=BLOCKSIZE,
        callback=input_callback
    ) as input_stream:

        # Beri waktu sedikit agar microphone aktif
        time.sleep(
            PLAYBACK_DELAY_SEC
        )

        with sd.OutputStream(
            samplerate=RECORD_SR,
            device=OUTPUT_DEVICE,
            channels=1,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=output_callback
        ) as output_stream:

            while not playback_done.is_set():
                sd.sleep(20)

    # --------------------------------------------------------
    # COMBINE RECORDED BLOCKS
    # --------------------------------------------------------

    if len(recorded_blocks) == 0:
        raise RuntimeError(
            "Tidak ada audio yang berhasil direkam."
        )

    recorded = np.concatenate(
        recorded_blocks,
        axis=0
    ).squeeze()

    # Potong agar panjang recording konsisten
    if len(recorded) > total_samples:
        recorded = recorded[:total_samples]

    return recorded.astype(
        np.float32
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_recording(audio):
    """
    QC awal saat acquisition.
    QC final tetap dijalankan menggunakan script QC Week 8.
    """

    metrics = calculate_audio_metrics(
        audio
    )

    if not metrics["finite"]:
        return (
            False,
            "NaN/Inf detected",
            metrics
        )

    if metrics["rms"] < SILENCE_THRESHOLD:
        return (
            False,
            (
                f"RMS terlalu kecil "
                f"({metrics['rms']:.8f})"
            ),
            metrics
        )

    if metrics["peak"] >= CLIP_THRESHOLD:
        return (
            False,
            (
                f"Clipping terdeteksi "
                f"(peak={metrics['peak']:.6f})"
            ),
            metrics
        )

    if metrics["clip_ratio"] > 0:
        return (
            False,
            (
                "Clipping samples terdeteksi "
                f"(ratio={metrics['clip_ratio']:.8f})"
            ),
            metrics
        )

    return (
        True,
        "",
        metrics
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print("WEEK 8 CONTROLLED REAL REPLAY RECORDING")
    print("==============================================")
    print()

    # --------------------------------------------------------
    # CHECK PLAN
    # --------------------------------------------------------

    if not PLAN_PATH.exists():
        raise FileNotFoundError(
            f"Plan tidak ditemukan: {PLAN_PATH}"
        )

    df = pd.read_csv(
        PLAN_PATH
    )

    print(
        "Plan:",
        PLAN_PATH
    )

    print(
        "Total jobs:",
        len(df)
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required = [
        "recording_id",
        "source_file_id",
        "playback_path",
        "recording_path",
        "recording_status",
        "repetition"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}"
        )

    # --------------------------------------------------------
    # VALIDATE WEEK 8
    # --------------------------------------------------------

    if len(df) != 240:
        raise ValueError(
            f"Week 8 seharusnya memiliki 240 jobs, "
            f"tetapi ditemukan {len(df)}."
        )

    if not (df["repetition"] == 2).all():
        raise ValueError(
            "Week 8 harus seluruhnya repetition = 2."
        )

    if df["recording_id"].duplicated().any():
        raise ValueError(
            "Ada duplicate recording_id."
        )

    # --------------------------------------------------------
    # PREPARE COLUMNS
    # --------------------------------------------------------

    text_columns = [
        "recording_status",
        "record_error",
        "record_subtype",
        "qc_status",
        "qc_reason",
        "agc_status"
    ]

    for col in text_columns:
        ensure_object_column(
            df,
            col
        )

    numeric_columns = [
        "record_peak",
        "record_rms",
        "record_clip_ratio",
        "record_duration_s",
        "record_sr",
        "speaker_level",
        "mic_level",
        "noise_floor",
        "qc_sr",
        "qc_duration_s",
        "qc_peak",
        "qc_rms",
        "qc_clip_ratio"
    ]

    for col in numeric_columns:
        ensure_numeric_column(
            df,
            col
        )

    if "qc_finite" not in df.columns:
        df["qc_finite"] = ""

    # --------------------------------------------------------
    # FIX ACQUISITION SETTINGS
    # --------------------------------------------------------

    # Setting yang sudah berhasil di Week 7.
    # Pastikan setting Windows aktual memang tetap 50/70.

    df["record_sr"] = RECORD_SR
    df["record_subtype"] = RECORD_SUBTYPE

    df["speaker_level"] = 50.0
    df["mic_level"] = 70.0

    # Hanya benar jika AGC memang dinonaktifkan
    df["agc_status"] = "off"

    # --------------------------------------------------------
    # FIND JOBS
    # --------------------------------------------------------

    status_lower = (
        df["recording_status"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    # done tidak direkam ulang.
    #
    # pending / failed akan diproses.
    jobs = df.index[
        status_lower != "done"
    ].tolist()

    already_done = int(
        (status_lower == "done").sum()
    )

    print(
        "Already done:",
        already_done
    )

    print(
        "Remaining:",
        len(jobs)
    )

    if len(jobs) == 0:
        print()
        print(
            "Semua recording Week 8 sudah DONE."
        )
        return

    print()
    print(
        f"Maksimal recording baru pada run ini: "
        f"{MAX_NEW_RECORDINGS}"
    )
    print()

    # --------------------------------------------------------
    # DEVICE INFORMATION
    # --------------------------------------------------------

    print(
        "Input device :",
        INPUT_DEVICE,
        sd.query_devices(
            INPUT_DEVICE
        )["name"]
    )

    print(
        "Output device:",
        OUTPUT_DEVICE,
        sd.query_devices(
            OUTPUT_DEVICE
        )["name"]
    )

    print(
        "Sample rate  :",
        RECORD_SR
    )

    print(
        "Subtype      :",
        RECORD_SUBTYPE
    )

    print()
    input(
        "Pastikan speaker=50 dan mic=70. "
        "Tekan ENTER untuk mulai..."
    )

    # --------------------------------------------------------
    # RECORD LOOP
    # --------------------------------------------------------

    successful_this_run = 0

    for idx in jobs:

        if successful_this_run >= MAX_NEW_RECORDINGS:
            break

        row = df.loc[idx]

        recording_id = str(
            row["recording_id"]
        )

        source_id = str(
            row["source_file_id"]
        )

        playback_path = Path(
            str(row["playback_path"])
        )

        recording_path = Path(
            str(row["recording_path"])
        )

        print()
        print(
            "=================================================="
        )

        print(
            f"Recording ID : {recording_id}"
        )

        print(
            f"Source ID    : {source_id}"
        )

        if "label" in df.columns:
            print(
                f"Label        : {row['label']}"
            )

        if "room" in df.columns:
            print(
                f"Room         : {row['room']}"
            )

        if "playback" in df.columns:
            print(
                f"Playback     : {row['playback']}"
            )

        if "mic" in df.columns:
            print(
                f"Mic          : {row['mic']}"
            )

        if "distance" in df.columns:
            print(
                f"Distance     : {row['distance']} m"
            )

        if "angle" in df.columns:
            print(
                f"Angle        : {row['angle']} degree"
            )

        print(
            f"Repetition   : {row['repetition']}"
        )

        print(
            f"Playback     : {playback_path}"
        )

        print(
            f"Output       : {recording_path}"
        )

        print(
            "=================================================="
        )

        input(
            "Siapkan kondisi fisik sesuai metadata. "
            "Tekan ENTER untuk record..."
        )

        try:

            # ------------------------------------------------
            # LOAD PLAYBACK
            # ------------------------------------------------

            playback_audio = load_playback_audio(
                playback_path
            )

            expected_duration = (
                len(playback_audio)
                / RECORD_SR
            )

            print(
                f"Playback duration: "
                f"{expected_duration:.3f} s"
            )

            print(
                "Recording..."
            )

            # ------------------------------------------------
            # RECORD
            # ------------------------------------------------

            recorded = play_and_record(
                playback_audio
            )

            duration = (
                len(recorded)
                / RECORD_SR
            )

            # ------------------------------------------------
            # QC AWAL
            # ------------------------------------------------

            valid, reason, metrics = (
                validate_recording(
                    recorded
                )
            )

            print()
            print(
                f"Peak       : "
                f"{metrics['peak']:.6f}"
            )

            print(
                f"RMS        : "
                f"{metrics['rms']:.6f}"
            )

            print(
                f"Clip ratio : "
                f"{metrics['clip_ratio']:.8f}"
            )

            print(
                f"Finite     : "
                f"{metrics['finite']}"
            )

            print(
                f"Duration   : "
                f"{duration:.3f} s"
            )

            # ------------------------------------------------
            # UPDATE METADATA
            # ------------------------------------------------

            df.at[
                idx,
                "record_peak"
            ] = metrics["peak"]

            df.at[
                idx,
                "record_rms"
            ] = metrics["rms"]

            df.at[
                idx,
                "record_clip_ratio"
            ] = metrics["clip_ratio"]

            df.at[
                idx,
                "record_duration_s"
            ] = duration

            df.at[
                idx,
                "record_sr"
            ] = RECORD_SR

            df.at[
                idx,
                "record_subtype"
            ] = RECORD_SUBTYPE

            if not valid:

                print()
                print(
                    "RECORDING FAILED:"
                )

                print(
                    reason
                )

                df.at[
                    idx,
                    "recording_status"
                ] = "failed"

                df.at[
                    idx,
                    "record_error"
                ] = reason

                # Tidak simpan recording gagal
                if recording_path.exists():
                    recording_path.unlink()

            else:

                # --------------------------------------------
                # SAVE WAV
                # --------------------------------------------

                recording_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                sf.write(
                    recording_path,
                    recorded,
                    RECORD_SR,
                    subtype=RECORD_SUBTYPE
                )

                df.at[
                    idx,
                    "recording_status"
                ] = "done"

                df.at[
                    idx,
                    "record_error"
                ] = ""

                # QC final belum dijalankan
                df.at[
                    idx,
                    "qc_status"
                ] = "NOT_CHECKED"

                print()
                print(
                    "RECORDING SUCCESS"
                )

                print(
                    f"Saved: {recording_path}"
                )

                successful_this_run += 1

            # ------------------------------------------------
            # SAVE MANIFEST AFTER EVERY JOB
            # ------------------------------------------------

            df.to_csv(
                PLAN_PATH,
                index=False
            )

            print()
            print(
                "Progress this run:",
                successful_this_run,
                "/",
                MAX_NEW_RECORDINGS
            )

        except KeyboardInterrupt:

            print()
            print(
                "Recording dihentikan oleh user."
            )

            df.to_csv(
                PLAN_PATH,
                index=False
            )

            break

        except Exception as e:

            error_text = str(e)

            print()
            print(
                "ERROR:"
            )

            print(
                error_text
            )

            df.at[
                idx,
                "recording_status"
            ] = "failed"

            df.at[
                idx,
                "record_error"
            ] = error_text

            df.to_csv(
                PLAN_PATH,
                index=False
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    df.to_csv(
        PLAN_PATH,
        index=False
    )

    status_lower = (
        df["recording_status"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    done = int(
        (status_lower == "done").sum()
    )

    failed = int(
        (status_lower == "failed").sum()
    )

    pending = len(df) - done - failed

    print()
    print(
        "=============================================="
    )

    print(
        "WEEK 8 RECORDING SUMMARY"
    )

    print(
        "=============================================="
    )

    print(
        "Recorded this run :",
        successful_this_run
    )

    print(
        "Total DONE        :",
        done
    )

    print(
        "Total FAILED      :",
        failed
    )

    print(
        "Remaining/PENDING :",
        pending
    )

    print(
        "Total jobs        :",
        len(df)
    )

    print()
    print(
        "Manifest saved:",
        PLAN_PATH
    )

    if done == 240:
        print()
        print(
            "SEMUA 240 RECORDING WEEK 8 SUDAH SELESAI."
        )

        print(
            "Langkah berikutnya: jalankan QC Week 8."
        )

    else:
        print()
        print(
            "Jalankan script ini kembali untuk "
            "melanjutkan recording."
        )


if __name__ == "__main__":
    main()