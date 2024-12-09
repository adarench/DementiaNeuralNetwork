import pandas as pd
import os
import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def extract_features(audio_path):
    try:
        # load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # extract audio
        duration = librosa.get_duration(y=y, sr=sr)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        
        # combine all features into a single array
        features = np.hstack([duration, mfccs, chroma, mel, contrast])
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_row(row):
    audio_file = row["Audio File"]
    
    # skip if the path is NaN or invalid
    if pd.isna(audio_file) or not isinstance(audio_file, str):
        return None
    
    # check if the file exists
    if not os.path.isfile(audio_file):
        return None
    
    print(f"Processing audio file: {audio_file}")
    
    # extract features
    features = extract_features(audio_file)
    
    if features is not None:
        return {
            "Participant": row.get("Participant", "Unknown"),
            "Group": row.get("Group", "Unknown"),
            "Task": row.get("Task", "Unknown"),
            "Audio File": audio_file,
            "Features": features
        }
    return None

def process_batch(batch):
    results = []
    for _, row in batch.iterrows():
        result = process_row(row)
        if result:
            results.append(result)
    return results

def main():
    # metadata
    metadata_file = "data/processed/metadata.csv" 
    metadata = pd.read_csv(metadata_file)
    
    metadata.rename(columns={"MP3 File": "Audio File"}, inplace=True)

    if "Audio File" not in metadata.columns:
        raise KeyError("The required column 'Audio File' is missing in the CSV.")

    # split into batches
    batch_size = 50
    batches = [metadata.iloc[i:i + batch_size] for i in range(0, len(metadata), batch_size)]

    # process pool for parallel batch processing
    extracted_features = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for future in futures:
            results = future.result()
            if results:
                extracted_features.extend(results)

    features_df = pd.DataFrame(extracted_features)

    output_file = "extracted_audio_features.csv" 
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

if __name__ == '__main__':
    main()
