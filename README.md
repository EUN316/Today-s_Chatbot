# :house: Today-s_Chatbot
---

오늘의 집 데이터를 활용한 리뷰 기반 Q&A Framework model<br>
사용자가 문의를 남기면 해당 문의의 답변으로 적합한 Q&A와 다른 사용자들이 작성한 리뷰를 채택하여 실시간 대화형으로 응답


## :vertical_traffic_light: Description
---

```
ㄴChatbot
ㄴData
  ㄴBase
  ㄴQnA
  ㄴReview
ㄴModel
  ㄴModel
```

### 1)Crawling.ipynb

'오늘의 집' 상품, QnA, Review 데이터 수집


2)Preprocessing_QA.ipynb & 2)Preprocessing_Review.ipynb  

  QnA, Review 데이터 맞춤법 검사 및 전처리 
    
    
  - 3)Mecab_QnA.ipynb & 3)Mecab_review.ipynb

    QnA, Review 데이터 형태소 분석기 Mecab 적용
  - 4)KoBERT_QnA.ipynb

    KoBERT를 사용하여 pretrained model 생성
  - 5)Embedding_Question.ipynb & 5)Embedding_Review.ipynb
  
    pretrained model을 적용하여 Embedding vector 생성 및 Classification
    
    Classification을 통한 Review Labeling
   
  - 6)Matching_System.ipynb
    
    입력받은 문의에 적합한 Q&A 2개와 Review 5개를 응답으로 제공하는 시스템 구현
    Sentiment Analysis, Word Count를 적용하여 Customize를 통한 결과 도출 가능
    
    
## Chatbot

  - Model을 Chatbot으로 구현
  - 문의와 상품번호를 입력받아 pretrained model로 입력받은 문의를 Embedding, Labeling 후 Cosine similarity를 계산하여 Threshold 이상인 Q&A 2개와 Review 5개 반환


## :speaker: Usage

---

### Requirements
```python
pip install kss freeze konlpy glounnlp pytorch transformers

cd Model
git clone https://github.com/SKTBrain/KoBERT # KoBERT
git clone https://github.com/ssut/py-hanspell # py-hanspell
```

### Installation


```python
git clone https://github.com/EUN316/Today-s_Chatbot.git
cd Today-s_Chatbot
```

### Matching System

```python
cd Model
# 1-5 ipynb 다 실행 > Data & Model 생성
# 6)Matching_System.ipynb 실행
```

### Chatbot

```python
# Chatbot > Base.yaml: path & paramter 수정 가능
# Chatbot > templates > index.html: Desktop & Mobile ver 수정 가능
cd Chatbot
python app1.py
```

## :clipboard: Demo
---
![데모사진](https://user-images.githubusercontent.com/55127132/136147109-fe8edf4a-3b70-4dfc-88b1-1242885eac45.png)


## :speech_balloon: Contributors

---

<table>
  <tr>
    <td align="center"><a href="https://github.com/eun723"><br /><sub><b>Kim Eun</b></sub></td>
    <td align="center"><a href="https://github.com/hyz218"><br /><sub><b>Yoon HyeJung</b></sub></td>
    <td align="center"><a href="https://github.com/EUN316"><br /><sub><b>Lee JungEun</b></sub></td>
  </tr>
</table>



## :pencil2: Reference

---

[SKT KoBERT](https://github.com/SKTBrain/KoBERT)

[Kochat](https://github.com/hyunwoongko/kochat)

[py-hanspell](https://github.com/ssut/py-hanspell)

