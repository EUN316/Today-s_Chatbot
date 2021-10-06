import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def Semantic_Search_Question(cfg:dict, question, input_embedding, input_class, product_id):
    """
    입력받은 질문과 기존 질문 중 같은 카테고리에 속한 질문간의 cosine similarity 진행
    """
    # path
    qna_path = cfg["qna_path"]

    # load file
    df = pd.read_pickle(qna_path+'question_embedding_SN_new.pkl')
    
    # 같은 상품 내의 질문만 추출
    df = df[df['product_id'] == product_id]
    df.reset_index(drop=True, inplace=True)

    # input_class와 같은 label만 추출
    label_idx = np.array(np.where(np.array(df['question_class'].tolist()) == input_class)).tolist()[0]
    df_in_input = df.loc[label_idx]
    df_in_input.reset_index(drop=True, inplace=True)
    
    # 질문 embedding
    question_embedding = df_in_input['question_embedding']

    # make tensor
    corpus_embeddings = torch.tensor(question_embedding)
    query_embedding = torch.tensor(input_embedding)
    
    # cosine similarity 계산
    cos_scores = cosine_similarity(query_embedding, corpus_embeddings)
    cos_socres_round = np.array(list(map(lambda x:int(x*1000)/1000, cos_scores[0]))) 
    df_in_input['input_question'] = np.repeat(question, len(df_in_input))
    df_in_input['score'] = cos_socres_round
    
    # 0.95 이상만 반환
    df_in_input_top = df_in_input[df_in_input['score']>=0.95]
    df_in_input_top.sort_values(by='score', ascending=False, inplace=True)
    df_in_input_top.reset_index(drop=True, inplace=True)
    
    return df_in_input_top


def Semantic_Search_Review(cfg:dict, question, input_embedding, input_class, product_id):
    """
    입력받은 질문과 리뷰 중 같은 카테고리에 속하거나 상품에 속하는 리뷰간의 cosine similarity 진행
    """
    # path
    review_path = cfg["review_path"]

    # load review data
    df = pd.read_pickle(review_path+'review_'+str(product_id)+'_embedding.pkl')
    
    # input_class와 같은 label만 추출
    label_idx = np.array(np.where(np.array(df['review_class'].tolist()) == input_class)).tolist()[0]
    
    if len(label_idx) != 0: # 같은 label의 리뷰가 하나라도 있으면
        df_in_input = df.loc[label_idx]
        df_in_input['using_label'] = input_class
        df_in_input.reset_index(drop=True, inplace=True)    
    else: # 하나도 없으면
        return pd.DataFrame() # 빈 데이터 프레임 반환
    
    # 리뷰 embedding
    review_embedding = df_in_input['review_embedding']
              
    # make tensor
    corpus_embeddings = torch.tensor(review_embedding)
    query_embedding = torch.tensor(input_embedding)
    
    # cosine similarity 계산
    cos_scores = cosine_similarity(query_embedding, corpus_embeddings)
    cos_socres_round = np.array(list(map(lambda x:int(x*1000)/1000, cos_scores[0]))) 
    df_in_input.loc[:,'input_question'] = np.repeat(question, len(df_in_input))
    df_in_input['score'] = cos_socres_round
    
    # 0.95 이상만
    df_in_input_top = df_in_input[df_in_input['score']>=0.95]
    df_in_input_top.sort_values(by='score', ascending=False, inplace=True)
    df_in_input_top.reset_index(drop=True, inplace=True)
    
    return df_in_input_top


def Text_Sentiment_Review(cfg:dict, df):
    """
    감정 분석 결과를 활용하여 긍정에 가까운 리뷰를 반환
    """
    # path
    data_path = cfg["data_path"]

    # sentiment table 불러오기
    with open(data_path+'table.pickle', 'rb') as file:
        table = pickle.load(file)
 
    # 초기화
    df['neg'] = 0
    df['neut'] = 0
    df['pos'] = 0

    # df 리뷰 전체 리스트
    review_list = df['comment_mecab'].apply(lambda x:x.replace('\n','').split(' ')).values.tolist()

    # table에 있는 경우만 남기기
    review_list = [[word for word in review if word in table] for review in review_list]

    # 리스트 내 word를 table 내 숫자로 바꾸기
    neg_list = [[float(table[word]['Neg']) for word in review] for review in review_list]
    pos_list = [[float(table[word]['Pos']) for word in review] for review in review_list]

    # 값 sum()
    neg = [sum(c) for c in neg_list]
    pos = [sum(c) for c in pos_list]

    # df 문장 단어 개수
    df['count'] = df['comment_mecab'].apply(lambda x:len(x.replace('\n','').split(' ')))

    # df에 구한 값 추가 & 개수로 나누기
    df['neg'] = round(neg/df['count'], 3)
    df['pos'] = round(pos/df['count'], 3)

    # 부정 점수 0.3 이하 & 긍정 점수 0.3 이상인 경우만 추출
    df = df[(df['neg']<=0.3)&(df['pos']>=0.3)]
    
    # column drop
    df_return = df.drop(columns=['count'])

    return df_return


def word_count_in_wordcloud_Review(cfg:dict, df):
    """
    word cloud로 구한 각 카테고리별 빈도수가 높은 상위 30개의 단어를 가지고 관련 단어가 나오는 개수를 변수로 생성
    """
    # path
    data_path = cfg["data_path"]

    # wordcloud dict 불러오기   
    with open(data_path+'word_dict.pickle', 'rb') as file:
        word_dict = pickle.load(file)
    
    # 전체 단어 count 개수
    df['word_count'] = df['comment_mecab'].apply(lambda x: len(x.split(' ')))
    
    # 예측된 label의 wordcloud에 포함된 단어
    df['word_count_in_cloud'] = 0
    for i in range(len(df)):
        df['word_count_in_cloud'][i] = len([w for w in df['comment_mecab'][i].replace('\n','').split(' ') if w in(word_dict[df['review_class'][i]][:30])]) # 30개만 불러오기
    
    # column 제거
    df_return = df.drop(columns=['word_count_in_cloud'])
    
    return df_return


def select_top2_Question(df):
    """
    선정된 유사한 질문를 기준에 맞춰 정렬하여 상위 k(<=2)개 반환
    """
    # score 기준 내림차순으로 정렬
    df.sort_values(by=['score'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True) # reset index
        
    if df.shape[0] < 2: # 개수가 2개보다 적으면
        df_return = df
    else:
        df_return = df[:2]

    # 필요한 column만 반환
    df_return = df_return[['input_question', 'score', 'question', 'answer']]

    return df_return

def select_top5_Review(df, sentiment, wordcount):    
    """
    선정된 유사한 리뷰를 기준에 맞춰 정렬하여 상위 k(<=5)개 반환
    """
    # Custom 여부에 따라 정렬
    if (sentiment == False) and (wordcount == False):
        sort_list = ['score', 'praise_count', 'image']
        df.sort_values(by=sort_list, ascending=False, inplace=True)
        
    if (sentiment == True) and (wordcount == False):
        sort_list = ['score', 'pos', 'neg', 'praise_count', 'image']
        df.sort_values(by=sort_list, ascending=[False, False, True, False, False], inplace=True)
    
    if (sentiment == False) and (wordcount == True):
        sort_list = ['score', 'word_count', 'praise_count', 'image']
        df.sort_values(by=sort_list, ascending=False, inplace=True)
    
    if (sentiment == True) and (wordcount == True):
        sort_list = ['score', 'pos', 'neg', 'word_count', 'praise_count', 'image']
        df.sort_values(by=sort_list, ascending=[False, False, True, False, False, False], inplace=True)  
        
    # reset index
    df.reset_index(drop=True, inplace=True)
    
    if df.shape[0] < 5: # 5개보다 적으면
        df_return = df
    else:
        df_return = df[:5]
    
    # 필요한 column만 추출
    out_list = ['prod_id', 'prod_name', 'input_question', 'comment'] + sort_list
    df_return = df_return[out_list]
    
    return df_return