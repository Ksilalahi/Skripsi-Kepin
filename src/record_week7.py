from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

PLAN_PATH = Path(
    "manifests/week7_real_replay_plan.csv"
)

RECORD_SR = 48000

# Device yang sebelumnya berhasil saat pilot.
# Periksa kembali jika device index Windows berubah.
INPUT_DEVICE = 1
OUTPUT_DEVICE = 3

# Pre-roll dan post-roll untuk merekam kondisi sebelum/sesudah playback.
PRE_ROLL = 0.5
POST_ROLL = 0.5

BLOCKSIZE = 1024

# Preliminary QC
SILENCE_THRESHOLD = 1e-4
CLIP_THRESHOLD = 0.99

# Maksimal recording BARU per satu kali menjalankan program.
MAX_NEW_RECORDINGS = 30


# ============================================================
# PLAYBACK + RECORDING
# ============================================================

def play_and_record_stream(
    playback_path,
    output_path
):

    # ========================================================
    # LOAD PLAYBACK MASTER
    # ========================================================

    playback_path = Path(
        playback_path
    )

    output_path = Path(
        output_path
    )

    if not playback_path.exists():

        raise FileNotFoundError(
            f"Playback master tidak ditemukan: "
            f"{playback_path}"
        )


    playback, sr = sf.read(
        playback_path,
        always_2d=True,
        dtype="float32"
    )


    if sr != RECORD_SR:

        raise ValueError(
            f"Playback bukan {RECORD_SR} Hz. "
            f"Ditemukan {sr} Hz pada "
            f"{playback_path}"
        )


    # ========================================================
    # PREPARE OUTPUT CHANNEL
    # ========================================================

    # Jika playback mono, duplikasi ke stereo
    # karena output device menggunakan 2 channel.
    if playback.shape[1] == 1:

        playback_out = np.repeat(
            playback,
            2,
            axis=1
        )

    else:

        playback_out = playback[
            :, :2
        ]


    playback_out = np.asarray(
        playback_out,
        dtype=np.float32
    )


    playback_frames = len(
        playback_out
    )


    playback_duration = (
        playback_frames
        /
        RECORD_SR
    )


    # Panjang rekaman final yang diinginkan:
    # pre-roll + audio + post-roll
    total_target_frames = int(
        (
            PRE_ROLL
            +
            playback_duration
            +
            POST_ROLL
        )
        *
        RECORD_SR
    )


    # ========================================================
    # BUFFERS
    # ========================================================

    recording_chunks = []

    playback_position = 0

    lock = threading.Lock()


    # ========================================================
    # INPUT CALLBACK
    # ========================================================

    def input_callback(
        indata,
        frames,
        time_info,
        status
    ):

        if status:

            print(
                "Input status:",
                status
            )


        recording_chunks.append(
            indata.copy()
        )


    # ========================================================
    # OUTPUT CALLBACK
    # ========================================================

    def output_callback(
        outdata,
        frames,
        time_info,
        status
    ):

        nonlocal playback_position


        if status:

            print(
                "Output status:",
                status
            )


        # Default output silence
        outdata.fill(0)


        with lock:

            start = (
                playback_position
            )

            end = min(
                start + frames,
                playback_frames
            )

            n = (
                end - start
            )


            if n > 0:

                outdata[
                    :n,
                    :
                ] = playback_out[
                    start:end,
                    :
                ]

                playback_position = (
                    end
                )


    # ========================================================
    # OPEN MICROPHONE
    # ========================================================

    # Input dibuka lebih dahulu supaya pre-roll
    # benar-benar ikut terekam.
    with sd.InputStream(
        device=INPUT_DEVICE,
        samplerate=RECORD_SR,
        channels=1,
        dtype="float32",
        blocksize=BLOCKSIZE,
        callback=input_callback
    ):

        # ====================================================
        # PRE-ROLL
        # ====================================================

        time.sleep(
            PRE_ROLL
        )


        # ====================================================
        # PLAYBACK
        # ====================================================

        with sd.OutputStream(
            device=OUTPUT_DEVICE,
            samplerate=RECORD_SR,
            channels=2,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=output_callback
        ):

            # Sedikit toleransi untuk memastikan
            # callback playback selesai.
            time.sleep(
                playback_duration
                +
                0.10
            )


        # ====================================================
        # POST-ROLL
        # ====================================================

        time.sleep(
            POST_ROLL
        )


    # ========================================================
    # COMBINE MICROPHONE DATA
    # ========================================================

    if len(
        recording_chunks
    ) == 0:

        raise RuntimeError(
            "Tidak ada audio yang diterima "
            "dari microphone."
        )


    recording = np.concatenate(
        recording_chunks,
        axis=0
    )


    # Callback bisa menghasilkan beberapa frame ekstra.
    # Potong sesuai target.
    if len(
        recording
    ) > total_target_frames:

        recording = recording[
            :total_target_frames
        ]


    # ========================================================
    # NUMERICAL CHECK
    # ========================================================

    if len(
        recording
    ) == 0:

        raise RuntimeError(
            "Rekaman kosong."
        )


    if not np.isfinite(
        recording
    ).all():

        raise RuntimeError(
            "Rekaman mengandung NaN atau Inf."
        )


    # ========================================================
    # PRELIMINARY QC
    # ========================================================

    peak = float(
        np.max(
            np.abs(
                recording
            )
        )
    )


    rms = float(
        np.sqrt(
            np.mean(
                recording ** 2
            )
        )
    )


    clip_ratio = float(
        np.mean(
            np.abs(
                recording
            )
            >=
            CLIP_THRESHOLD
        )
    )


    duration_s = float(
        len(recording)
        /
        RECORD_SR
    )


    # Silence
    if rms < SILENCE_THRESHOLD:

        raise RuntimeError(
            "Rekaman terlalu kecil / silence. "
            f"RMS={rms:.8f}"
        )


    # Clipping
    if clip_ratio > 0:

        raise RuntimeError(
            "Clipping terdeteksi. "
            f"peak={peak:.6f}, "
            f"clip_ratio={clip_ratio:.8f}"
        )


    # ========================================================
    # SAVE MASTER RECORDING
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    sf.write(
        output_path,
        recording,
        RECORD_SR,
        subtype="PCM_24"
    )


    return {
        "peak": peak,
        "rms": rms,
        "clip_ratio": clip_ratio,
        "duration_s": duration_s
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CHECK MANIFEST
    # ========================================================

    if not PLAN_PATH.exists():

        raise FileNotFoundError(
            f"Plan tidak ditemukan: "
            f"{PLAN_PATH}"
        )


    df = pd.read_csv(
        PLAN_PATH
    )


    # ========================================================
    # FIX PANDAS COLUMN DTYPES
    #
    # Sangat penting:
    # kolom kosong dalam CSV sering dibaca Pandas sebagai
    # float64 karena nilainya NaN.
    #
    # Tanpa bagian ini:
    # df.at[idx, "record_error"] = ""
    #
    # dapat menghasilkan:
    # TypeError: Invalid value '' for dtype 'float64'
    # ========================================================

    text_columns = [
        "recording_status",
        "record_error",
        "qc_status",
        "speaker_level",
        "mic_level",
        "noise_floor",
        "agc_status"
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
        "record_peak",
        "record_rms",
        "record_clip_ratio",
        "record_duration_s"
    ]


    for col in numeric_columns:

        if col not in df.columns:

            df[col] = np.nan

        else:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required = [
        "recording_id",
        "source_file_id",
        "label",
        "room",
        "playback",
        "mic",
        "distance",
        "angle",
        "repetition",
        "playback_path",
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
            "Kolom week7 plan tidak lengkap: "
            +
            ", ".join(
                missing
            )
        )


    # ========================================================
    # HEADER
    # ========================================================

    print()
    print(
        "======================================================="
    )

    print(
        "MINGGU 7 - CONTROLLED REAL REPLAY ACQUISITION"
    )

    print(
        "======================================================="
    )


    done_before = (
        df[
            "recording_status"
        ]
        .astype(str)
        .str.lower()
        .eq("done")
        .sum()
    )


    failed_before = (
        df[
            "recording_status"
        ]
        .astype(str)
        .str.lower()
        .eq("failed")
        .sum()
    )


    print(
        "Total planned recordings :",
        len(df)
    )

    print(
        "Already DONE             :",
        done_before
    )

    print(
        "Currently FAILED         :",
        failed_before
    )

    print(
        "Remaining                :",
        len(df)
        -
        done_before
    )

    print(
        "Maximum new this run     :",
        MAX_NEW_RECORDINGS
    )


    # ========================================================
    # AUDIO DEVICE CHECK
    # ========================================================

    print()
    print(
        "Checking audio devices..."
    )


    try:

        input_info = sd.query_devices(
            INPUT_DEVICE
        )

        output_info = sd.query_devices(
            OUTPUT_DEVICE
        )


        print(
            "Input device :",
            INPUT_DEVICE,
            "-",
            input_info["name"]
        )


        print(
            "Output device:",
            OUTPUT_DEVICE,
            "-",
            output_info["name"]
        )


        print(
            "Sample rate  :",
            RECORD_SR
        )


        sd.check_input_settings(
            device=INPUT_DEVICE,
            channels=1,
            samplerate=RECORD_SR,
            dtype="float32"
        )


        sd.check_output_settings(
            device=OUTPUT_DEVICE,
            channels=2,
            samplerate=RECORD_SR,
            dtype="float32"
        )


    except Exception as e:

        raise RuntimeError(
            "Audio device tidak siap. "
            f"Detail: {e}"
        )


    print(
        "Audio device check: PASS"
    )


    # ========================================================
    # COUNTER BATCH
    # ========================================================

    new_recordings = 0


    # ========================================================
    # RECORDING LOOP
    # ========================================================

    for idx, row in df.iterrows():

        # ====================================================
        # STOP OTOMATIS SETELAH 30 REKAMAN BARU
        # ====================================================

        if (
            new_recordings
            >=
            MAX_NEW_RECORDINGS
        ):

            break


        recording_id = str(
            row[
                "recording_id"
            ]
        )


        playback_path = Path(
            str(
                row[
                    "playback_path"
                ]
            )
        )


        output_path = Path(
            str(
                row[
                    "output_path"
                ]
            )
        )


        status = str(
            row[
                "recording_status"
            ]
        ).strip().lower()


        # ====================================================
        # SAFE RESUME
        # ====================================================

        # Hanya skip apabila:
        # status DONE dan file WAV benar-benar ada.
        if (
            status == "done"
            and
            output_path.exists()
        ):

            print(
                f"{recording_id}: "
                "SKIP (already done)"
            )

            continue


        # Manifest mengatakan done tetapi WAV hilang.
        # Reset otomatis menjadi pending.
        if (
            status == "done"
            and
            not output_path.exists()
        ):

            print()
            print(
                f"{recording_id}: WARNING"
            )

            print(
                "Status = done tetapi WAV tidak ditemukan."
            )

            print(
                "Status di-reset menjadi pending."
            )


            df.at[
                idx,
                "recording_status"
            ] = "pending"


            df.at[
                idx,
                "record_error"
            ] = ""


            df.to_csv(
                PLAN_PATH,
                index=False
            )


        # ====================================================
        # CHECK PLAYBACK MASTER
        # ====================================================

        if not playback_path.exists():

            print()
            print(
                f"{recording_id}: FAILED"
            )

            print(
                "Playback master tidak ditemukan:"
            )

            print(
                playback_path
            )


            df.at[
                idx,
                "recording_status"
            ] = "failed"


            df.at[
                idx,
                "record_error"
            ] = "playback_file_not_found"


            df.to_csv(
                PLAN_PATH,
                index=False
            )


            continue


        # ====================================================
        # DISPLAY RECORDING CONFIGURATION
        # ====================================================

        print()
        print(
            "======================================================="
        )

        print(
            f"RECORDING: {recording_id}"
        )

        print(
            "======================================================="
        )


        print(
            "Source ID  :",
            row[
                "source_file_id"
            ]
        )


        print(
            "Label      :",
            row[
                "label"
            ]
        )


        print(
            "Room       :",
            row[
                "room"
            ]
        )


        print(
            "Playback   :",
            row[
                "playback"
            ]
        )


        print(
            "Mic        :",
            row[
                "mic"
            ]
        )


        print(
            "Distance   :",
            row[
                "distance"
            ]
        )


        print(
            "Angle      :",
            row[
                "angle"
            ]
        )


        print(
            "Repetition :",
            row[
                "repetition"
            ]
        )


        print(
            "Playback path:"
        )

        print(
            playback_path
        )


        print(
            "Output path:"
        )

        print(
            output_path
        )


        print(
            "======================================================="
        )


        # ====================================================
        # MANUAL CONFIRMATION
        # ====================================================

        command = input(
            "\nAtur konfigurasi fisik sesuai metadata.\n"
            "Tekan ENTER untuk mulai recording.\n"
            "Ketik q lalu ENTER untuk berhenti: "
        )


        if (
            command
            .strip()
            .lower()
            ==
            "q"
        ):

            print()
            print(
                "Recording dihentikan oleh user."
            )

            break


        # ====================================================
        # RECORD
        # ====================================================

        try:

            print()
            print(
                "Recording..."
            )


            metrics = (
                play_and_record_stream(
                    playback_path,
                    output_path
                )
            )


            # =================================================
            # UPDATE MANIFEST
            # =================================================

            df.at[
                idx,
                "recording_status"
            ] = "done"


            df.at[
                idx,
                "record_peak"
            ] = metrics[
                "peak"
            ]


            df.at[
                idx,
                "record_rms"
            ] = metrics[
                "rms"
            ]


            df.at[
                idx,
                "record_clip_ratio"
            ] = metrics[
                "clip_ratio"
            ]


            df.at[
                idx,
                "record_duration_s"
            ] = metrics[
                "duration_s"
            ]


            # Kolom ini sekarang object,
            # sehingga string kosong aman.
            df.at[
                idx,
                "record_error"
            ] = ""


            new_recordings += 1


            print()
            print(
                "PRELIMINARY QC: PASS"
            )


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
                f"Duration   : "
                f"{metrics['duration_s']:.3f} s"
            )


            print(
                "Saved       :",
                output_path
            )


        except Exception as e:

            # =================================================
            # RECORDING FAILED
            # =================================================

            error_message = str(
                e
            )


            df.at[
                idx,
                "recording_status"
            ] = "failed"


            # Aman karena record_error sudah dtype object.
            df.at[
                idx,
                "record_error"
            ] = error_message


            print()
            print(
                f"{recording_id}: FAILED"
            )


            print(
                "Reason:",
                error_message
            )


        # ====================================================
        # SAVE MANIFEST AFTER EVERY JOB
        # ====================================================

        df.to_csv(
            PLAN_PATH,
            index=False
        )


        # ====================================================
        # CURRENT PROGRESS
        # ====================================================

        completed = (
            df[
                "recording_status"
            ]
            .astype(str)
            .str.lower()
            .eq("done")
            .sum()
        )


        failed = (
            df[
                "recording_status"
            ]
            .astype(str)
            .str.lower()
            .eq("failed")
            .sum()
        )


        print()
        print(
            "---------------------------------------"
        )

        print(
            "Progress total         :",
            f"{completed}/{len(df)}"
        )


        print(
            "New recordings this run:",
            f"{new_recordings}/"
            f"{MAX_NEW_RECORDINGS}"
        )


        print(
            "Failed currently       :",
            failed
        )

        print(
            "---------------------------------------"
        )


    # ========================================================
    # FINAL SAVE
    # ========================================================

    df.to_csv(
        PLAN_PATH,
        index=False
    )


    # ========================================================
    # FINAL COUNTS
    # ========================================================

    completed = (
        df[
            "recording_status"
        ]
        .astype(str)
        .str.lower()
        .eq("done")
        .sum()
    )


    failed = (
        df[
            "recording_status"
        ]
        .astype(str)
        .str.lower()
        .eq("failed")
        .sum()
    )


    pending = (
        df[
            "recording_status"
        ]
        .astype(str)
        .str.lower()
        .isin(
            [
                "",
                "pending",
                "nan"
            ]
        )
        .sum()
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "======================================================="
    )

    print(
        "BATCH RECORDING SELESAI"
    )

    print(
        "======================================================="
    )


    print(
        "Recording baru batch ini :",
        new_recordings
    )


    print(
        "Total DONE               :",
        completed
    )


    print(
        "Total FAILED             :",
        failed
    )


    print(
        "Total PENDING            :",
        pending
    )


    print(
        "Target Minggu 7          :",
        "240"
    )


    print(
        "Manifest                 :",
        PLAN_PATH
    )


    print()


    # ========================================================
    # NEXT ACTION
    # ========================================================

    if (
        new_recordings
        >=
        MAX_NEW_RECORDINGS
    ):

        print(
            "STATUS:"
        )

        print(
            "Batch 30 recording selesai."
        )

        print(
            "JANGAN lanjut batch berikutnya dahulu."
        )

        print(
            "Jalankan Quality Control (QC) "
            "untuk batch ini."
        )


    elif (
        completed
        >=
        len(df)
    ):

        print(
            "STATUS:"
        )

        print(
            "Seluruh recording Minggu 7 "
            "telah selesai."
        )


    else:

        print(
            "STATUS:"
        )

        print(
            "Program berhenti sebelum "
            "30 recording baru selesai."
        )

        print(
            "Program dapat dijalankan kembali "
            "untuk melanjutkan recording."
        )


    print(
        "======================================================="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()