import pandas as pd
from transformers import pipeline
from tqdm import tqdm  # Import tqdm for progress bar
from datasets import Dataset, DatasetDict
import multiprocessing

# Set the start method to 'spawn' to fix CUDA in multiprocessing issue
multiprocessing.set_start_method('spawn', force=True)

# Load dataset (TSV format)
df = pd.read_csv('datasets/train.tsv', sep='\t', header=0, quoting=3)

# Separate by class (balanced extraction)
df_0 = df[df["label"] == 0]  # Non-toxic
df_1 = df[df["label"] == 1]  # Toxic

# Extract 10% for Finnish (balanced)
n_finnish_0 = int(0.10 * len(df_0))
n_finnish_1 = int(0.10 * len(df_1))
finnish_samples_0 = df_0.sample(n=n_finnish_0, random_state=42)
finnish_samples_1 = df_1.sample(n=n_finnish_1, random_state=42)
finnish_samples = pd.concat([finnish_samples_0, finnish_samples_1]).reset_index(drop=True)

# Extract 25% for German (balanced)
n_german_0 = int(0.25 * len(df_0))
n_german_1 = int(0.25 * len(df_1))
german_samples_0 = df_0.sample(n=n_german_0, random_state=42)
german_samples_1 = df_1.sample(n=n_german_1, random_state=42)
german_samples = pd.concat([german_samples_0, german_samples_1]).reset_index(drop=True)

# Load translation models (GPU enabled)
translator_fi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fi", device=0)  # Use GPU (device=0)
translator_de = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de", device=0)  # Use GPU (device=0)

# Convert the dataframes to HuggingFace datasets for better processing
finnish_samples = Dataset.from_pandas(finnish_samples)
german_samples = Dataset.from_pandas(german_samples)

# Batch processing using the dataset API
def translate_batch(batch, translator):
    translated_texts = translator(batch['text'])
    batch['text'] = [t['translation_text'] for t in translated_texts]
    return batch

# Translate to Finnish (batch processing)
finnish_samples = finnish_samples.map(lambda batch: translate_batch(batch, translator_fi), batched=True, batch_size=32)

# Translate to German (batch processing)
german_samples = german_samples.map(lambda batch: translate_batch(batch, translator_de), batched=True, batch_size=32)

# Add ids for the translated samples
finnish_samples = pd.DataFrame(finnish_samples)
german_samples = pd.DataFrame(german_samples)

finnish_samples["id"] = ["fin_test_" + str(i) for i in range(len(finnish_samples))]
german_samples["id"] = ["ger_test_" + str(i) for i in range(len(german_samples))]

# Keep only necessary columns
finnish_samples = finnish_samples[["id", "text", "label"]]
german_samples = german_samples[["id", "text", "label"]]
df = df[["id", "text", "label"]]

# Merge datasets
augmented_df = pd.concat([df, finnish_samples, german_samples])

# Save as TSV
augmented_df.to_csv("datasets/augmented_train.tsv", sep='\t', index=False, quoting=3)

print("Translation and augmentation complete! 🚀")
