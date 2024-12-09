import os
import pandas as pd

# paths
transcript_path = "data/raw/transcripts/"
audio_path = "data/raw/audio/"

# init list
metadata = []

# traverse dementia and control subdirectories
for group in ["dementia", "control"]:
    group_transcripts = os.path.join(transcript_path, group)
    group_audio = os.path.join(audio_path, group)
    
    for task in os.listdir(group_transcripts):  
        task_transcripts = os.path.join(group_transcripts, task)
        task_audio = os.path.join(group_audio, task)
        
        if not os.path.isdir(task_transcripts):
            continue
        
        for filename in os.listdir(task_transcripts):
            if filename.endswith(".cha"):
                participant_id = filename.split(".")[0]
                cha_file = os.path.join(task_transcripts, filename)
                mp3_file = os.path.join(task_audio, f"{participant_id}.mp3")
                
                mp3_exists = os.path.exists(mp3_file)
                
                metadata.append({
                    "Participant": participant_id,
                    "Group": group,
                    "Task": task,
                    "CHA File": cha_file,
                    "MP3 File": mp3_file if mp3_exists else None,
                    "MP3 Exists": mp3_exists
                })

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv("data/processed/metadata.csv", index=False)

print(metadata_df.head())
