#!/usr/bin/env python
# coding: utf-8

from keybert import KeyBERT
import torch
from datasets import load_dataset
import os
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, AutoModelForMaskedLM
import pandas as pd
import numpy as np
import collections

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Loading pretraining data
text_dataset = load_dataset('text', data_files={'pretrain': 'summarized_data.txt'})

# Getting texts
doc = text_dataset["pretrain"]["text"]

# Initializing KeyBERT
kw_model = KeyBERT(model="bert-base-uncased")


# Extracting keywords from the given document
keywords = kw_model.extract_keywords(doc)
kw_extracted = kw_model.extract_keywords(doc,
                                         keyphrase_ngram_range=(1, 1), # Unigrams
                                         stop_words='english',
                                         use_mmr=True, # Using MMR to diversify keyword selection
                                         top_n=10, # Extracting up to top 10 keywords
                                         diversity=0.8) # MMR threshold


# Getting the keywords
keys = [item[0] for sublist in kw_extracted for item in sublist]

# find frequency of detected keywords
frequency = collections.Counter(keys)

# Save keywords in a dictionary with their frequency
keywords_with_frequency = dict(sorted(frequency.items(), key=lambda item: item[1], reverse=True))

# Sort keywords based on their frequency in decreasing order
new_list = sorted(frequency, key=frequency.get, reverse=True)


# Saving a set of keywords in a text file
with open('most-frequent-keywords.txt', 'w') as f:
    for item in new_list:
        f.write("%s\n" % item)

# Saving keywords with their frequency in a dictionary-like format in a text file
with open('most-frequent-keywords-frequency.txt', 'w') as f:
    for key, value in keywords_with_frequency.items():
        f.write("%s:%s\n" %(key, value))

