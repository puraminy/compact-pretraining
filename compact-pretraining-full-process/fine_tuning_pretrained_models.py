#!/usr/bin/env python
# coding: utf-8

import os
from datasets import load_dataset
import torch
from transformers import (AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, recall_score)
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_labels = "Replace with the number of labels."
model_ckpt = "Replace with the address of the pretrained model."

# Loading datasets
data = load_dataset("csv", data_files={'train': 'train.csv', 'valid': 'valid.csv', 'test': 'test.csv'})

# Initializing tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', model_max_length=512)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

# Tokenizing datasets
tokenized_data = data.map(tokenize, batched=True, batch_size=None)

# Initialzing model
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)

# Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# Training arguments
training_args = TrainingArguments(output_dir = "Replace with your desired output directory", 
                                  num_train_epochs=4,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=16, 
                                  per_device_eval_batch_size=16,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch", 
                                  push_to_hub=False,
                                  log_level="error",
                                  save_strategy="epoch", 
                                  load_best_model_at_end=True, 
                                  save_total_limit=1,
                                  metric_for_best_model='f1',
                                  logging_dir="Replace with your desired output directory", 
                                  logging_strategy="epoch",
                                  dataloader_num_workers=20)

# Initializing trainer
trainer = Trainer(model=model,
                  args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=tokenized_data["train"],
                  eval_dataset=tokenized_data["valid"])


# Start fine-tuning
trainer.train()

# See the performance of model on test set
preds_output = trainer.predict(tokenized_data["test"])
print(preds_output.metrics)
