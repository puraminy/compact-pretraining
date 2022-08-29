#!/usr/bin/env python
# coding: utf-8


from transformers import pipeline, set_seed, AutoTokenizer
from datasets import load_dataset
import os

# Setting up the seed and the visible GPU
set_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Note: 'device=0' enables using GPU by summarization pipeline. This argument doesn't specify which GPU to use.
# To specify which GPU to use, you should use 'os.environ["CUDA_VISIBLE_DEVICES"]="0"' as above.
pipe = pipeline("summarization", model="facebook/bart-large-cnn", max_length=1024, batch_size=8, device=0)

# Loading the pretraining data
text_dataset = load_dataset('text', data_files={'pretrain': 'your_pretraining_data.txt'})

# Setting up the summarization pipeline
pipe_out = pipe(text_dataset['pretrain']['text'])

# Saving all the summaries in a single text file.
with open(r'summarized_data.txt', 'w') as fp:
    for item in pipe_out:
        fp.write("%s\n" % item['summary_text'])
    print('Done')
