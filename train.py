import sys
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

from src.preprocessing import *
from src.models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

gc.collect()
torch.cuda.empty_cache()

bert_name = "bert-base-uncased"
path = "datasets/"
tokenizer = AutoTokenizer.from_pretrained(bert_name)

train_dataloader, val_dataloader = load_and_processing(path, tokenizer)

print("###data loading complete###")
print("length of train_dataloader:", len(train_dataloader))


epochs = 10


# Initialize the model
model = BERTClassifier(bert_name, 2).to(device)
# model.load_state_dict(torch.load('bert_pt_classifier.pth'))


optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * epochs
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)


def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)





# Open the file in write mode (or append mode, depending on your preference)
with open('training_output.txt', 'w') as f:
    # Redirect stdout to the file
    sys.stdout = f

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train(model, train_dataloader, optimizer, scheduler, device)
        accuracy, report = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)

        torch.save(model.state_dict(), f"bert_pt_classifier_{epoch+1}.pth")

    # Reset stdout back to the default (console)
    sys.stdout = sys.__stdout__

print("Training output saved to 'training_output.txt'")