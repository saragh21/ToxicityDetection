import pandas as pd

# Load your raw train set
df = pd.read_csv(r"C:\Users\sarag\OneDrive\Desktop\excercise&homework\semester4\snlp\project\Toxicity-Recognition-Kaggle-2024\data\train_2025.tsv", sep='\t', quoting=3)

# Separate by class
toxic_df = df[df['label'] == 1]
non_toxic_df = df[df['label'] == 0]

# Target count = number of non-toxic samples
target_count = len(non_toxic_df)

# Oversample toxic to match target count
df_toxic_oversampled = toxic_df.sample(n=target_count, replace=True, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([non_toxic_df, df_toxic_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Confirm class balance
print("Before balancing:")
print(df['label'].value_counts())

print("After oversampling:")
print(balanced_df["label"].value_counts())
# print("Class counts after balancing:")
# print(balanced_df['label'].value_counts())

# Save (optional)
balanced_df.to_csv(r"C:\Users\sarag\OneDrive\Desktop\excercise&homework\semester4\snlp\project\Toxicity-Recognition-Kaggle-2024\data\balanced_train.tsv", sep='\t', index=False, quoting=3)
# Count duplicates in toxic samples
duplicate_count = df_toxic_oversampled.duplicated(subset=["text"]).sum()
print(f"Number of duplicate toxic comments: {duplicate_count}") 
dupes = df_toxic_oversampled[df_toxic_oversampled.duplicated(subset=["text"], keep=False)]
print(dupes.sample(5))
 