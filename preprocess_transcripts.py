import pylangacq
import os
import pandas as pd
import re

metadata_path = "data/processed/metadata.csv"
metadata = pd.read_csv(metadata_path)

features = []


def extract_linguistic_features(cha_file):
   
    with open(cha_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    participant_id = "*PAR:"
    participant_utterances = []

    for line in lines:
        if line.startswith(participant_id):
           
            utterance = line.split(":", 1)[1].strip()
           
            clean_utterance = re.sub(r"\[[^]]*\]|<[^>]*>|&[a-z]+|[+][^ ]*", "", utterance)
            participant_utterances.append(clean_utterance)

    if not participant_utterances:
        return None  

    total_words = sum(len(utt.split()) for utt in participant_utterances)
    unique_words = len(set(word for utt in participant_utterances for word in utt.split()))
    mean_sentence_length = total_words / len(participant_utterances) if participant_utterances else 0
    disfluencies = sum(utt.lower().count("uh") + utt.lower().count("um") for utt in participant_utterances)

    return {
        "Total Words": total_words,
        "Unique Words": unique_words,
        "Mean Sentence Length": mean_sentence_length,
        "Disfluencies": disfluencies
    }


if __name__ == '__main__':
  
    for _, row in metadata.iterrows():
        cha_file = row["CHA File"]
        if os.path.exists(cha_file):
            participant_id = row["Participant"]
            group = row["Group"]
            task = row["Task"]
            
            try:
           
                linguistic_features = extract_linguistic_features(cha_file)
                if linguistic_features:
                    features.append({
                        "Participant": participant_id,
                        "Group": group,
                        "Task": task,
                        **linguistic_features
                    })
            except Exception as e:
                print(f"Error processing {cha_file}: {e}")

    features_df = pd.DataFrame(features)
    features_df.to_csv("data/processed/linguistic_features.csv", index=False)

    print("Linguistic features saved to data/processed/linguistic_features.csv")
