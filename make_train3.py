import pandas as pd
from pathlib import Path

data_path = Path('.', 'data/sentiment140')  # 데이터 폴더 경로
train_file = data_path / 'train.csv'
train3_file = data_path / 'train3.csv'

# CSV 읽기
train = pd.read_csv(train_file, header=None, encoding='latin1')

# train.csv에서 랜덤으로 5000개의 데이터만 저장
train3 = train.sample(5000, random_state=1)

# CSV로 저장
train3.to_csv(train3_file, index=False, header=False, encoding='latin1')

print(f"train3.csv saved with {len(train3)} rows.")
