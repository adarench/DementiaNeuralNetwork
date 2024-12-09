import pandas as pd
import numpy as np

file_path = "data/features/combined_features.csv" 
df = pd.read_csv(file_path)

columns_to_drop = ['Group_x', 'Group_y', 'Task_x', 'Task_y', 'MP3 Exists', 'CHA File', 'MP3 File']
df = df.drop(columns=columns_to_drop, errors='ignore')

df.rename(columns={'Audio File': 'AudioFile', 'Features': 'FeatureArray'}, inplace=True)

df = df[df['AudioFile'].notna()] 

def expand_features(feature_array):
    try:
        feature_list = eval(feature_array)  
        return pd.Series(feature_list)
    except:
        return pd.Series([np.nan] * 5)  

feature_cols = ['Feature_' + str(i) for i in range(1, 6)] 
df[feature_cols] = df['FeatureArray'].apply(expand_features)

df.drop(columns=['FeatureArray'], inplace=True)

group_counts = df['Group'].value_counts()
print("Group Counts Before Balancing:")
print(group_counts)

min_count = group_counts.min()
df_balanced = df.groupby('Group').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

cleaned_file_path = "cleaned_dataset.csv"
df_balanced.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to {cleaned_file_path}")

print("Cleaned Dataset Summary:")
print(df_balanced.info())
print(df_balanced.head())
