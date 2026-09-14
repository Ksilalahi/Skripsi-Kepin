from pathlib import Path
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf


INPUT_DEVICE = 1
OUTPUT_DEVICE = 3


def play_and_record(
    input_path,
    output_path,
    rec_sr=48000
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD PLAYBACK AUDIO
    # ========================================================

    audio, sr = sf.read(
        input_path,
        always_2d=True,
        dtype="float32"
    )

    if sr != rec_sr:
        raise ValueError(
            f"Playback harus {rec_sr} Hz, "
            f"tetapi file adalah {sr} Hz"
        )

    duration = len(audio) / rec_sr

    print("Playback :", input_path)
    print("Recording:", output_path)

    print(
        "Input device :",
        sd.query_devices(INPUT_DEVICE)["name"]
    )

    print(
        "Output device:",
        sd.query_devices(OUTPUT_DEVICE)["name"]
    )

    print(
        f"Durasi playback: {duration:.2f} detik"
    )

    # ========================================================
    # RECORDING DIBUAT LEBIH PANJANG
    # ========================================================

    pre_roll = 0.5
    post_roll = 0.5

    record_duration = (
        duration
        + pre_roll
        + post_roll
    )

    record_frames = int(
        record_duration * rec_sr
    )

    recording_holder = {}

    # ========================================================
    # INPUT STREAM
    # ========================================================

    def record_worker():

        with sd.InputStream(
            device=INPUT_DEVICE,
            samplerate=rec_sr,
            channels=1,
            dtype="float32"
        ) as input_stream:

            recording, overflowed = (
                input_stream.read(
                    record_frames
                )
            )

            recording_holder["data"] = (
                recording.copy()
            )

            recording_holder["overflowed"] = (
                overflowed
            )

    # ========================================================
    # START MICROPHONE
    # ========================================================

    thread = threading.Thread(
        target=record_worker
    )

    thread.start()

    # microphone aktif lebih dahulu
    time.sleep(pre_roll)

    # ========================================================
    # OUTPUT STREAM
    # ========================================================

    output_channels = audio.shape[1]

    with sd.OutputStream(
        device=OUTPUT_DEVICE,
        samplerate=rec_sr,
        channels=output_channels,
        dtype="float32"
    ) as output_stream:

        output_stream.write(audio)

    # ========================================================
    # WAIT RECORDING
    # ========================================================

    thread.join()

    if "data" not in recording_holder:
        raise RuntimeError(
            "Recording gagal."
        )

    recording = recording_holder["data"]

    # ========================================================
    # QC
    # ========================================================

    peak = float(
        np.max(
            np.abs(recording)
        )
    )

    rms = float(
        np.sqrt(
            np.mean(
                recording ** 2
            )
        )
    )

    print()
    print("Recording statistics")
    print("--------------------")
    print(f"Peak : {peak:.6f}")
    print(f"RMS  : {rms:.6f}")

    if recording_holder.get(
        "overflowed",
        False
    ):
        print(
            "WARNING: input overflow."
        )

    if peak < 1e-4:
        print(
            "WARNING: rekaman sangat kecil."
        )

    if peak >= 0.99:
        print(
            "WARNING: kemungkinan clipping."
        )

    # ========================================================
    # SAVE MASTER RECORDING
    # ========================================================

    sf.write(
        output_path,
        recording,
        rec_sr,
        subtype="PCM_24"
    )

    print()
    print(
        "Selesai:"
    )

    print(
        output_path
    )


if __name__ == "__main__":

    play_and_record(
        "data/real_replay/playback_master/test_playback.wav",
        "data/real_replay/pilot/test_stream.wav"
    )