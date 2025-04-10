import pandas as pd
import re
import stanza
import stopwordsiso as stopwords_iso
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import os

# Set custom paths
custom_stanza_dir = os.path.expanduser('~/scratch/stanza_resources')
os.environ['STANZA_RESOURCES_DIR'] = custom_stanza_dir

nltk_data_path = os.path.expanduser("~/scratch/nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download stopwords if not available
try:
    nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# Initialize Stanza pipelines
stanza.download('en')
stanza.download('de')
stanza.download('fi')

nlp_en = stanza.Pipeline(lang='en', processors='tokenize,mwt,lemma', use_gpu=True)
nlp_de = stanza.Pipeline(lang='de', processors='tokenize,mwt,lemma', use_gpu=True)
nlp_fi = stanza.Pipeline(lang='fi', processors='tokenize,mwt,lemma', use_gpu=True)

# Load stopwords
stop_words_en = set(stopwords.words('english'))
stop_words_de = set(stopwords.words('german'))
stop_words_fi = stopwords_iso.stopwords("fi")

# Preprocessing function
def preprocess(text, lang):
    text = re.sub(r'[^\w\s]', '', str(text))
    if lang == 'en':
        nlp = nlp_en
        stop_words = stop_words_en
    elif lang == 'de':
        nlp = nlp_de
        stop_words = stop_words_de
    elif lang == 'fi':
        nlp = nlp_fi
        stop_words = stop_words_fi
    else:
        return text

    result = [word.lemma.lower()
              for token in nlp(text).iter_tokens()
              for word in token.words
              if word.lemma.lower() not in stop_words]
    return " ".join(result)

# Detect language from ID
def detect_lang(row):
    if str(row['id']).startswith("fin_"):
        return "fi"
    elif str(row['id']).startswith("ger_"):
        return "de"
    else:
        return "en"

# Preprocess a single file
def preprocess_file(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path, sep='\t')
    tqdm.pandas(desc=f"Processing {os.path.basename(path)}")
    df['lang'] = df.apply(detect_lang, axis=1)
    df['preprocessed_text'] = df.progress_apply(lambda row: preprocess(row['text'], row['lang']), axis=1)
    return df

# Process datasets and save output with _preprocessed suffix
datasets = ['augmented_train.tsv', 'dev_2025.tsv', 'test_2025.tsv']
for file in datasets:
    processed_df = preprocess_file(file)
    out_path = file.replace(".tsv", "_preprocessed.tsv")
    processed_df.to_csv(out_path, sep='\t', index=False)
    print(f"✅ Saved preprocessed data to {out_path}")
