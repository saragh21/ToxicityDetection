import torch
from torch import nn
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm  # Import tqdm for the progress bar
from src.models import BERTClassifier
import sys  # Needed to exit the script

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model from the .pth file
bert_name = "bert-base-uncased"
model = BERTClassifier(bert_name, 2).to(device)
tokenizer = AutoTokenizer.from_pretrained(bert_name)

model.load_state_dict(torch.load('bert_pt_classifier_10.pth', map_location=device))
model.eval()

# Function for sentiment prediction
def predict_sentiment(text, model, tokenizer, device, max_length=256):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs  # Unpack if tuple
        _, preds = torch.max(logits, dim=1)
    return int(preds.item())

# Load test dataset
test_df = pd.read_csv('datasets/test.tsv', sep='\t', header=0, quoting=3)

# check if its the right length
if(len(test_df['id'])!= 12791):
    print(f"Error: Expected 12791 rows, but got {len(test_df['id'])}. Exiting.")
    sys.exit(1)  # Exit with error status    

# Perform predictions using a loop with tqdm for progress bar
predictions = []
for text in tqdm(test_df['text'], desc="Predicting", unit="sample"):
    predictions.append(predict_sentiment(text, model, tokenizer, device))

# Store predictions in dataframe
test_df['predicted'] = predictions

# Save results
output_df = test_df[['id', 'predicted']]
output_df.to_csv("results/test.tsv", sep='\t', index=False, header=True, quoting=3)

print('Prediction complete')
