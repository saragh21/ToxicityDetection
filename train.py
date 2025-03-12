import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.nn.functional as F
from tqdm import tqdm
import os
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
import gc
import re
import emoji
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")


# load raw data
train_df = pd.read_csv('datasets/train.tsv', sep='\t')
dev_df = pd.read_csv('datasets/dev.tsv', sep='\t')
test_df = pd.read_csv('datasets/test.tsv', sep='\t')

# data cleaning
def clean_text(text):
    if not isinstance(text, str):  # Ensure input is a string
        return ""

    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters (except basic punctuation)
    text = re.sub(r"[^A-Za-z0-9.,!?'\"]+", ' ', text)

    return text


# Apply text cleaning to the dataset
train_df['text'] = train_df['text'].apply(clean_text)
dev_df['text'] = dev_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)


train_texts = train_df['text'].tolist()
val_texts = dev_df['text'].tolist()

train_labels = train_df['label'].values
val_labels = dev_df['label'].values


class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt',
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True,
                                  return_token_type_ids=False)

        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)}
    
max_length = 256
batch_size = 16

# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create Dataset objects
train_dataset = ToxicDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = ToxicDataset(val_texts, val_labels, tokenizer, max_length)

# Create DataLoader objects
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

# Initialize the model
model = BERTClassifier("bert-base-uncased", 2).to(device)


# Freeze all layers except the classifier
for param in model.bert.parameters():
    param.requires_grad = False

# Keep only the classification head trainable
for param in model.fc.parameters():
    param.requires_grad = True

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")



def train_model(model, train_loader, val_loader, epochs=3, accumulation_steps=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # Validation
        model.eval()
        val_loss = 0.0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation', leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch + 1}/{epochs} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)


train_model(model, train_dataloader, val_dataloader)
