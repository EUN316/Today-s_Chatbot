import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

from models.KoBERT.kobert.utils import get_tokenizer
from models.KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


class BERTDataset(Dataset):
    """
    make dataset using BERT
    """
    def __init__(self, dataset, sent_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i]) for i in dataset] # sent_idx를 제거
        
    def __getitem__(self, i):
        return (self.sentences[i])

    def __len__(self):
        return (len(self.sentences))
    
    
class BERTClassifier(nn.Module):
    """
    BERTClassifier
    """
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=5,  ##### 수정
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        
        classifier = self.classifier(pooler)
        
        if self.dr_rate:
            out = self.dropout(classifier)
  
        return pooler, out