#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
import os
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import (TrainingArguments, Trainer, AutoModelForMaskedLM)
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Loading pretraining data
text_dataset = load_dataset('text', data_files={'train': 'pretraining_data.txt'})

# Initializing tokenizer
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch['text'],
                     truncation=True,
                     max_length=512,
                     return_special_tokens_mask=True)

# Tokenizing data
tokenized_data = text_dataset.map(tokenize, batched=True)

# Data collator (dataloader) setting
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


# Training arguments setting
training_args = TrainingArguments(output_dir = "Replace with your desired output directory",
                                  per_device_train_batch_size=16,
                                  logging_strategy="epoch",
                                  save_strategy="epoch",
                                  num_train_epochs=2,
                                  log_level="error", 
                                  save_total_limit=1,
                                  report_to="none",
                                  lr_scheduler_type = 'linear')

# Trainer setting
trainer = Trainer(model=AutoModelForMaskedLM.from_pretrained(model_ckpt),
                  tokenizer=tokenizer,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=tokenized_data["train"])

# Start pretraining
trainer.train()

