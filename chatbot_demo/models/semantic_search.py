import torch
import numpy as np
import pandas as pd


def pytorch_cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def Semantic_Search_Question(cfg:dict, question, input_embedding, input_class, product_id):
    """
    입력받은 질문과 기존 질문 중 같은 카테고리에 속한 질문간의 cosine similarity 진행
    """
    # path
    qna_path = cfg["qna_path"]

    # load file
    df = pd.read_csv(qna_path+'question_embedding.csv', sep='\t')

    # 같은 상품에서만
    df = df[df['product_id']==int(product_id)]
    df.reset_index(drop=True, inplace=True)

    # input_class와 같은 label만 추출
    label_idx = np.array(np.where(np.array(df['question_class'].tolist()) == input_class)).tolist()[0]
    df_in_input = df.loc[label_idx]
    df_in_input.reset_index(drop=True, inplace=True)
    # 질문 embedding
    df_in_input['question_embedding'] = df_in_input['question_embedding'].str.replace('[','').str.replace(']','').apply(lambda x:list(map(float, x.split(','))))
    question_embedding = np.array(df_in_input['question_embedding'].tolist())
    # 기존 label
    question_class = np.array(df_in_input['question_class'].tolist())

    # make tensor
    corpus_embeddings = torch.tensor(question_embedding)
    query_embedding = torch.tensor(input_embedding)

    # cosine similarity
    cos_scores = pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    # 0.99 이상만
    top_results = np.where(cos_scores >= 0.9)
    df_in_input_top = df_in_input.loc[top_results[0]]

    for idx in top_results[0]:
        df_in_input_top.loc[idx, 'input_question'] = question
        df_in_input_top.loc[idx, 'score'] = cos_scores.numpy()[idx]

    df_in_input_top.reset_index(drop=True, inplace=True)

    return df_in_input_top


def Semantic_Search_Review(cfg:dict, question, input_embedding, input_class, product_id):
    """
    입력받은 질문과 리뷰 중 같은 카테고리에 속하거나 상품에 속하는 리뷰간의 cosine similarity 진행
    """
    # path
    review_path = cfg["review_path"]

    # load review data
    df = pd.read_csv(review_path+'review_'+product_id+'_embedding.csv', sep=',')

    # input_class와 같은 label만 추출
    label_idx = np.array(np.where(np.array(df['review_class'].tolist()) == input_class)).tolist()[0]

    if len(label_idx) != 0: # 하나라도 있으면
        df_in_input = df.loc[label_idx]
        df_in_input['using_label'] = input_class
        df_in_input.reset_index(drop=True, inplace=True)
    else: # 하나도 없으면
        label_idx = np.array(np.where(np.array(df['review_class'].tolist()) == 1)).tolist()[0] # 상품 카테고리
        df_in_input = df.loc[label_idx]
        df_in_input['using_label'] = 1
        df_in_input.reset_index(drop=True, inplace=True)

    # 질문 embedding
    df_in_input['review_embedding'] = df_in_input['review_embedding'].str.replace('[','').str.replace(']','').apply(lambda x:list(map(float, x.split(','))))
    review_embedding = np.array(df_in_input['review_embedding'].tolist())
    # 예측된 class
    review_class = np.array(df_in_input['review_class'].tolist())

    # make tensor
    corpus_embeddings = torch.tensor(review_embedding)
    query_embedding = torch.tensor(input_embedding)

    # cosine similarity
    cos_scores = pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    # 0.9 이상만
    top_results = np.where(cos_scores >= 0.9)
    df_in_input_top = df_in_input.loc[top_results[0]]

    for idx in top_results[0]:
        df_in_input_top.loc[idx, 'input_question'] = question
        df_in_input_top.loc[idx, 'score'] = cos_scores.numpy()[idx]

    df_in_input_top.reset_index(drop=True, inplace=True)

    return df_in_input_top


def select_top1_Question(df):
    """
    선정된 유사한 질문를 기준에 맞춰 정렬하여 상위 k(<=5)개 반환
    """
    # score 높은순 & 날짜 최신순
    df.sort_values(by=['score','answer_time'], ascending=False, inplace=True)

    # reset index
    df.reset_index(drop=True, inplace=True)

    if df.shape[0] > 0:
        df_return = df[:1]

    # 필요한 내용만 추출
    df_return = df_return[['product_id', 'product_name', 'input_question', 'score', 'question', 'answer']]

    return df_return


def select_top5_Review(df):
    """
    선정된 유사한 리뷰를 기준에 맞춰 정렬하여 상위 k(<=5)개 반환
    """
    # score 높은순 & 추천 수 많은순 & image_url 여부
    df.sort_values(by=['score','praise_count','image_url'], ascending=False, inplace=True)

    # reset index
    df.reset_index(drop=True, inplace=True)

    if df.shape[0] < 5:
        df_return = df
    else:
        df_return = df[:5]

    # 필요한 내용만 추출
    df_return = df_return[['prod_id', 'prod_name', 'input_question', 'score', 'comment', 'praise_count', 'image_url']]

    return df_return
