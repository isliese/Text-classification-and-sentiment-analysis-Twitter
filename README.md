# Text-classification-and-sentiment-analysis-Twitter

A dataset that contains 1.6 million training and 350 test tweets from 2009 with algorithmically assigned binary positive and negative sentiment scores that are fairly evenly split.

Download the data from <a href="http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip">here.</a>

## 데이터 로딩 및 전처리
train: 1.6M 트윗, polarity 0(negative) 또는 4(positive) → polarity>0로 binarize → 0/1
test: 350 트윗, polarity 0 또는 4 → 같은 방식으로 0/1
140자 이상 트윗 제거 → 당시 트위터 제한 길이 반영
Parquet 저장: 빠른 입출력 위해 parquet 포맷 사용

<br>

## 데이터 탐색
train.polarity.value_counts() → positive/negative가 거의 균등하게 분포
트윗 길이 분포 (sns.distplot) → 대부분 140자 이하
train.user.nunique() → unique 사용자 수
train.user.value_counts().head(10) → 상위 트윗 사용자 확인
요약: 데이터가 충분히 다양하고, 일부 파워유저가 트윗을 많이 작성함

<br>

## Feature Extraction
CountVectorizer: 문서-단어 행렬 생성
min_df=0.001 → 전체 트윗 중 0.1% 미만 등장 단어 제거
max_df=0.8 → 너무 자주 등장하는 단어 제거
stop_words='english' → 불용어 제거
train_dtm / test_dtm → DTM(sparse matrix) 생성
요약: 각 트윗을 숫자 특징 벡터로 변환

<br>

## 모델 학습
Multinomial Naive Bayes 학습 (nb.fit(train_dtm, train.polarity))
Naive Bayes는 텍스트 분류에서 자주 쓰이는 간단하면서 강력한 모델

<br>

## 예측 및 평가
Test polarity 예측: predicted_polarity = nb.predict(test_dtm)
Accuracy: accuracy_score(test.polarity, predicted_polarity)
결과: NB 모델이 TextBlob보다 정확도가 높음

<br>

## TextBlob 활용
TextBlob은 rule-based sentiment 분석 → polarity [-1,1]
샘플 트윗 확인:
sample_positive → positive score
sample_negative → negative score
estimate_polarity(text) 함수로 전체 test set 평가
Accuracy: (test.sentiment>0).astype(int)
NB보다 낮음
요약: TextBlob은 통계적 모델이 아니라 단어 기반 polarity 추정 → ML 모델(NB)보다 덜 정확

<br>

## ROC / AUC 분석
roc_auc_score 비교:
NB predict_proba → test set AUC 높음
TextBlob → AUC 낮음
ROC curve 시각화:
좌: TextBlob polarity score 분포
우: NB vs TextBlob ROC curve 비교
결과: Naive Bayes 모델이 TextBlob보다 분류 성능이 좋음

<br>

## 결론
데이터 전처리: 140자 이하, binary polarity → 분석용으로 잘 준비됨
특징 벡터화: CountVectorizer → 문서-단어 행렬 성공
모델 비교:
Naive Bayes: 높은 정확도, AUC → ML 기반 분류기 우세
TextBlob: 단순 rule-based → 성능 NB보다 낮음
시각화:
긍정 트윗은 TextBlob polarity가 높음
ROC curve → NB가 TextBlob보다 더 민감/정확하게 positive/negative 구분
한 줄 요약:
Naive Bayes 모델이 Twitter 감성분석에서는 TextBlob보다 훨씬 정확하며, simple DTM 기반 ML 모델로도 충분히 좋은 성능을 보여줌.