import pandas as pd
from datasets import Dataset
import multiprocessing
from transformers import pipeline, MarianMTModel, MarianTokenizer
import torch

# Use this to fix CUDA + multiprocessing issues
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multiprocessing.set_start_method('spawn', force=True)
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de').to(device)
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# --- Load English data ---
df_en = pd.read_csv('datasets/balanced_train.tsv', sep='\t', quoting=3)
df_en = df_en[["id", "text", "label"]]
english_len = len(df_en)

# --- Load Finnish data ---
finnish_comments = pd.read_json(
    "hf://datasets/TurkuNLP/jigsaw_toxicity_pred_fi/train_fi_deepl.jsonl.bz2",
    lines=True
)
finnish_comments['label'] = (finnish_comments['label_toxicity'] > 0.5).astype(int)
finnish_comments['id'] = ['fin_' + str(i) for i in range(len(finnish_comments))]
finnish_comments = finnish_comments[['id', 'text', 'label']]

# Sample Finnish comments (10% of English, balanced)
fi_n = int(0.1 * english_len // 2)
fi_sample = pd.concat([ 
    finnish_comments[finnish_comments['label'] == 0].sample(fi_n, random_state=42),
    finnish_comments[finnish_comments['label'] == 1].sample(fi_n, random_state=42)
])

# --- Load German datasets ---
# GermEval21 (Train set)
german_df_train = pd.read_csv('datasets/GermEval21_Toxic_Train.csv')
german_df_train['label'] = german_df_train['Sub1_Toxic'].astype(int)
german_df_train['id'] = ['ger21_' + str(i) for i in range(len(german_df_train))]
german_df_train = german_df_train.rename(columns={'comment_text': 'text'})
german_df_train = german_df_train[['id', 'text', 'label']]

# GermEval21 (Test set)
german_df_test = pd.read_csv('datasets/GermEval21_Toxic_TestData.csv')
german_df_test['label'] = german_df_test['Sub1_Toxic'].astype(int)
german_df_test['id'] = ['ger21_' + str(i) for i in range(len(german_df_test))]
german_df_test = german_df_test.rename(columns={'comment_text': 'text'})
german_df_test = german_df_test[['id', 'text', 'label']]

# GermEval2018 (Train set)
splits = {'train': 'data/train-00000-of-00001.parquet'}
germeval2018_train = pd.read_parquet("hf://datasets/philschmid/germeval18/" + splits["train"])
germeval2018_train_df = pd.DataFrame({
    "id": ['ger18_' + str(i) for i in range(len(germeval2018_train))],
    "text": germeval2018_train["text"],
    "label": germeval2018_train["binary"].apply(lambda x: 1 if x.lower() == "offense" else 0)
})

# GermEval2018 (Test set)
splits = {'test': 'data/test-00000-of-00001.parquet'}
germeval2018_test = pd.read_parquet("hf://datasets/philschmid/germeval18/" + splits["test"])
germeval2018_test_df = pd.DataFrame({
    "id": ['ger18_' + str(i) for i in range(len(germeval2018_test))],
    "text": germeval2018_test["text"],
    "label": germeval2018_test["binary"].apply(lambda x: 1 if x.lower() == "offense" else 0)
})

# --- Combine German datasets (train + test) ---
german_full = pd.concat([german_df_train, german_df_test, germeval2018_train_df, germeval2018_test_df], ignore_index=True)
print(f"Total German rows after merging train and test sets: {len(german_full)}")

# --- Complete German dataset to 25% of English ---
target_de_n = int(0.25 * english_len)  # 25% of English comments
current_de_n = len(german_full)

if current_de_n < target_de_n:
    additional_needed = target_de_n - current_de_n
    print(f"Additional German samples needed: {additional_needed}")
    n_trans_ger = additional_needed // 2
    df_en_0 = df_en[df_en["label"] == 0]
    df_en_1 = df_en[df_en["label"] == 1]  # Fixed bug: this should be label == 1, not 0.
    trans_ger_samples_0 = df_en_0.sample(n=n_trans_ger, random_state=42)
    trans_ger_samples_1 = df_en_1.sample(n=n_trans_ger, random_state=42)
    trans_ger_samples = pd.concat([trans_ger_samples_0, trans_ger_samples_1]).reset_index(drop=True)

    translator_de = pipeline("translation_en_to_de", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)  # Use GPU (device=0)
    trans_ger_samples = Dataset.from_pandas(trans_ger_samples)

    def translate_batch(batch, translator):
        inputs = tokenizer(
            batch['text'],
            max_length=400,
            truncation=True,
            padding=True,
            return_tensors="pt",
            )
        
        inputs = {key: value.to(device) for key, value in inputs.items()}

        translated_outputs = translator.model.generate(**inputs)
        
        translated_texts = tokenizer.batch_decode(translated_outputs, skip_special_tokens=True)
        batch['text'] = translated_texts
        return batch

    # Use `map()` to batch process the dataset for translation
    trans_ger_samples = trans_ger_samples.map(lambda batch: translate_batch(batch, translator_de), batched=True, batch_size=16)

    trans_ger_samples = pd.DataFrame(trans_ger_samples)

    trans_ger_samples["id"] = ["ger_trans_" + str(i) for i in range(len(trans_ger_samples))]

    german_full = pd.concat([german_full, trans_ger_samples], ignore_index=True)
else:
    print("✅ German dataset already has 25% of the English comments. No additional data needed.")

# --- Combine all datasets ---
combined_df = pd.concat([df_en, fi_sample, german_full], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean text: flatten newlines
combined_df["text"] = combined_df["text"].apply(lambda x: x.replace("\n", " ").replace("\\n", " ") if isinstance(x, str) else x)

# Save final dataset
combined_df.to_csv(
    'datasets/augmented_train.tsv',
    sep='\t',
    index=False,
    quoting=3,
    escapechar='\\'
)

print("✅ Final dataset saved with balanced English, Finnish (10%), and German (25% of English) comments!")
