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

from models.model import BERTDataset, BERTClassifier

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def input_BERTClassifier(cfg:dict, que_mecab):
    """
    pre-traiend된 BERTClassifier 모델을 가져와서
    입력받은 질문에 대한 embedding & classification 진행
    """
    # path
    model_path = cfg['model_path']
    device = cfg['device']
    max_len = cfg['parameters']['max_len']
    batch_size = cfg['parameters']['batch_size']
    hidden_size = cfg['parameters']['hidden_size']
    num_classes = cfg['parameters']['num_classes']
    dr_rate = cfg['parameters']['dr_rate']
    
    # load model
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    
    # make dataloader
    input_data = BERTDataset(que_mecab, 0, tok, max_len, True, False)
    input_dataloader = torch.utils.data.DataLoader(input_data, batch_size=batch_size, num_workers=5)
    
    # load pre-trained KoBERTClassifier
    model = BERTClassifier(bertmodel, hidden_size, num_classes, dr_rate).to(device)
    model.load_state_dict(torch.load(model_path+'epoch_5_qna_sep_model_4.pt', map_location=torch.device(device)))
    model.eval() 
 
    # embedding & classification
    for batch_id, (token_ids, valid_length, segment_ids) in enumerate(input_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length  

        test_embedding, test_probability = model.forward(token_ids, valid_length, segment_ids)

        test_embedding = np.array(test_embedding.tolist())
        test_probability = np.array(test_probability.tolist())

        torch.cuda.empty_cache() # GPU 캐시 삭제
        
    # class 
    test_class = np.argmax(np.array(test_probability))
        
    return test_embedding, test_class