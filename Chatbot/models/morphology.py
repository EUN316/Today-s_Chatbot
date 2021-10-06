from konlpy.tag import Mecab


def make_question_mecab_tokens(question):
    """
    입력받은 질문에 대한 형태소 분석 진행
    1. mecab
    2. 조사 제거
    3. 한 문장으로 다시 결합
    """
    # load mecab
    mecab = Mecab()
    
    # mecab 돌리기
    que_mecab = mecab.pos(question[0])

    # 조사 제거
    morpheme = ['NNG','NNP','NNB','NNBC','NR','NP','VV','VA','VX','VCP','VCN','MM','MAG','MAJ','IC','SN']
    tmp = []
    que_tokens = []

    for t in que_mecab:
        if t[1] in morpheme:
            que_tokens.append(t[0])

    if len(que_tokens)==0:
        que_tokens.append('')

    # 한 문장으로 결합
    que_tokens_str = [' '.join(que_tokens)]
    
    return que_tokens_str