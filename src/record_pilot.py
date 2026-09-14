from pathlib import Path
import pandas as pd
from play_and_record import play_and_record

PLAN = "manifests/pilot_acquisition_plan.csv"

def main():
    df = pd.read_csv(PLAN)
    for i, row in df.iterrows():
        pilot_id = row["pilot_id"]
        playback_path = row[
            "playback_path"
        ]

        output_path = row[
            "output_path"
        ]

        status = row[
            "recording_status"
        ]

        if status == "done":
            print(
                f"{pilot_id} sudah direkam, skip."
            )
            continue

        print("\n================================")
        print(
            f"Recording {i+1}/{len(df)}"
        )
        print("Pilot ID :", pilot_id)
        print("Label    :", row["label"])
        print("Source   :", row["source_file_id"])
        print("Playback :", playback_path)
        print("Output   :", output_path)
        print("================================")

        input("\nTekan ENTER jika speaker dan " "microphone sudah siap...")
        try:
            play_and_record(
                playback_path,
                output_path,
                rec_sr=48000
            )

            df.loc[
                i,
                "recording_status"
            ] = "done"

            # langsung simpan progres
            df.to_csv(
                PLAN,
                index=False
            )

            print("Status: DONE")

        except Exception as e:
            df.loc[
                i,
                "recording_status"
            ] = "error"

            df.to_csv(PLAN,index=False)

            print("ERROR:",e)
            answer = input("Lanjut ke file berikutnya? [y/n]: ")
            if answer.lower() != "y":
                break

if __name__ == "__main__":
    main()