#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
import os
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, AutoModelForMaskedLM
import pandas as pd
from KeywordMasking import DataCollatorForKeywordMasking

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Loading pretraining data
text_dataset = load_dataset('text', data_files={'train': 'summarized_data.txt'})


# Initializing tokenizer
model_ckpt = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch["text"],
                     truncation=True,
                     max_length=512,
                     return_special_tokens_mask=True)

# Tokenizing data
tokenized_data = text_dataset.map(tokenize, batched=True)

# Reading the text file containing all top n keywords into a list
with open("all_of_of_the_keywords.txt") as f:
      text_file = f.read().splitlines()

keywords = []
for line in text_file:
    keywords.append(line)


# Initializing data collator for keyword masking
data_collator = DataCollatorForKeywordMasking(tokenizer=tokenizer,
                                              list_of_keywords=keywords,
                                              mlm_probability=0.75)

# Training arguments setting
training_args = TrainingArguments(output_dir = "Your desired output directory",
                                  per_device_train_batch_size=16, 
                                  logging_strategy="epoch",
                                  save_strategy="epoch",
                                  num_train_epochs=2,
                                  save_total_limit=1, 
                                  log_level="error",
                                  report_to="none",
                                  lr_scheduler_type='constant',
                                  dataloader_num_workers=16)

# Initializing trainer
trainer = Trainer(model=AutoModelForMaskedLM.from_pretrained(model_ckpt),
                 tokenizer=tokenizer,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=tokenized_data["train"])

# Start pretraining
trainer.train()
