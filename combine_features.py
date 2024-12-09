import pandas as pd

# load linguistic features
linguistic_df = pd.read_csv('data/features/linguistic_features.csv')

# load prosodic features
prosodic_df = pd.read_csv('data/features/extracted_audio_features.csv')

# merge on 'Participant'
combined_df = pd.merge(linguistic_df, prosodic_df, on='Participant')

# metadata
metadata_df = pd.read_csv('data/processed/metadata.csv')  # Example metadata file
final_df = pd.merge(combined_df, metadata_df, on='Participant')

# save the final combined dataset
output_path = 'data/features/combined_features.csv'
final_df.to_csv(output_path, index=False)
print(f"Combined dataset saved to {output_path}")
