import gc
import torch
import torch.nn as nn
from transformers import AutoModel


class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(pooled_output)
        return self.fc(x)
