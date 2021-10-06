# KoBERT models에 clone해온 후 파일 실행
from models.KoBERT.kobert.utils import get_tokenizer
from models.KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model

from models.morphology import make_question_mecab_tokens # mecab
from models.embedding import input_BERTClassifier # BERT embedding
from models.model import BERTDataset # BERT dataset
from models.semantic_search import Semantic_Search_Question, Semantic_Search_Review # Semantic search
from models.semantic_search import Text_Sentiment_Review, word_count_in_wordcloud_Review # Custom 
from models.semantic_search import select_top2_Question, select_top5_Review # select top5

import warnings
warnings.filterwarnings(action='ignore')

def QnA_system(cfg:dict, product_id, question):
    # Parameters
    sentiment = cfg["parameters"]["sentiment"]
    wordcount = cfg["parameters"]["wordcount"]

    # run mecab
    que_mecab = make_question_mecab_tokens(question)

    # input data embedding & class
    input_embedding, input_class = input_BERTClassifier(cfg, que_mecab)
    
    # Semantic Search between input & question
    question_top = Semantic_Search_Question(question, input_embedding, input_class, product_id)
    
    # select qna top2
    question_top2 = select_top2_Question(question_top)
    
    # Semantic Search between input & review
    review_top = Semantic_Search_Review(question, input_embedding, input_class, product_id)
    
    # Make sentiment column
    if sentiment == True: 
        review_top = Text_Sentiment_Review(review_top)
    # Make word count column
    if wordcount == True:
        review_top = word_count_in_wordcloud_Review(review_top)
    
    # select review top5
    review_top5 = select_top5_Review(review_top, sentiment, wordcount)  

    return question_top2, review_top5
