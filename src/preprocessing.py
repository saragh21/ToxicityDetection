import re
import string
import numpy as np
import pandas as pd
import emoji
from torch.utils.data import DataLoader

from data_loaders import *

def preprocess_dataset(values):
    new_values = list()
    # Emoticons
    emoticons = [':-)', ':)', '(:', '(-:', ':))', '((:', ':-D', ':D', 'X-D', 'XD', 'xD', 'xD', '<3', '</3', ':\*',
                 ';-)',
                 ';)', ';-D', ';D', '(;', '(-;', ':-(', ':(', '(:', '(-:', ':,(', ':\'(', ':"(', ':((', ':D', '=D',
                 '=)',
                 '(=', '=(', ')=', '=-O', 'O-=', ':o', 'o:', 'O:', 'O:', ':-o', 'o-:', ':P', ':p', ':S', ':s', ':@',
                 ':>',
                 ':<', '^_^', '^.^', '>.>', 'T_T', 'T-T', '-.-', '*.*', '~.~', ':*', ':-*', 'xP', 'XP', 'XP', 'Xp',
                 ':-|',
                 ':->', ':-<', '$_$', '8-)', ':-P', ':-p', '=P', '=p', ':*)', '*-*', 'B-)', 'O.o', 'X-(', ')-X']

    for value in values:
        # Remove dots
        text = value.replace(".", "").lower()
        text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
        users = re.findall("[@]\w+", text)
        for user in users:
            text = text.replace(user, "<user>")
        urls = re.findall(r'(https?://[^\s]+)', text)
        if len(urls) != 0:
            for url in urls:
                text = text.replace(url, "<url >")
        for emo in text:
            if emo in emoji.EMOJI_DATA:
                text = text.replace(emo, "<emoticon >")
        for emo in emoticons:
            text = text.replace(emo, "<emoticon >")
        numbers = re.findall('[0-9]+', text)
        for number in numbers:
            text = text.replace(number, "<number >")
        text = text.replace('#', "<hashtag >")
        text = re.sub(r"([?.!,¿])", r" ", text)
        text = "".join(l for l in text if l not in string.punctuation)
        text = re.sub(r'[" "]+', " ", text)
        new_values.append(text)
    return new_values


def load_and_processing(path_to_data, tokenizer, max_length, batch_size):
    # load raw data
    train_df = pd.read_csv(path_to_data + 'train.tsv', sep='\t')
    dev_df = pd.read_csv(path_to_data + 'dev.tsv', sep='\t')
    # test_df = pd.read_csv('datasets/test.tsv', sep='\t')

    train_texts = preprocess_dataset(train_df['text'].tolist())
    val_texts = preprocess_dataset(dev_df['text'].tolist())
    # test_texts = preprocess_dataset(test_df['text'].tolist())

    train_labels = train_df['label'].values
    val_labels = dev_df['label'].values


    train_dataset = ToxicDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ToxicDataset(val_texts, val_labels, tokenizer, max_length)

    # Create DataLoader objects
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader
