from models.KoBERT.kobert.utils import get_tokenizer
from models.KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model

from models.morphology import make_question_mecab_tokens # mecab
from models.embedding import input_BERTClassifier # BERT embedding
from models.model import BERTDataset # BERT dataset
from models.semantic_search import Semantic_Search_Question, Semantic_Search_Review # Semantic search
from models.semantic_search import select_top1_Question, select_top5_Review # select top5

import warnings
warnings.filterwarnings(action='ignore')

def QnA_system(cfg:dict, product_id, question):
    # run mecab
    que_mecab = make_question_mecab_tokens(question)
    # print('run mecab')

    # input data embedding & class
    input_embedding, input_class = input_BERTClassifier(cfg, que_mecab)
    # print('run embedding&classification')

    # Semantic Search between input & question
    question_top = Semantic_Search_Question(cfg, question, input_embedding, input_class, product_id)
    # select question top5
    question_top1 = select_top1_Question(question_top)
    # print('run semantic search about question')

    # Semantic Search between input & review
    review_top = Semantic_Search_Review(cfg, question, input_embedding, input_class, product_id)
    # select top5
    review_top5 = select_top5_Review(review_top)
    # print('run semantic search about review')

    # print(question_top5)
    # print(review_top5)

    # print('상품명: ', question_top5.product_id[0])
    # print('상품 번호: ', question_top5.product_name[0])
    # print('질문: ', question)
    #
    # for i in range(len(question_top5)):
    #     print('--------------------------')
    #     print('similarity: ', question_top5.score[i])
    #     print('질문: ', question_top5.question[i])
    #     print('답변: ', question_top5.answer[i])
    #
    # for i in range(len(review_top5)):
    #     print('--------------------------')
    #     print('리뷰: ', review_top5.comment[i])
    #     print('추천 수: ', review_top5.praise_count[i])
    #     print('image_url: ', review_top5.image_url[i])

    return question_top1, review_top5
