# ETRI-Human-understanding-AI


# Model Architecture


# Requirements
```
torch==
pyarrow==9.0.0
pandas==1.5.3
numpy==1.23.5
scipy==1.13.0
```

# 폴더 구성
원활한 실행을 위해 아래와 같은 구조의 폴더 구조로 만들어놓아야 함
`+-- \[디렉토리\]` 형태는 디렉토리 이름을 의미하고, `- 파일명`은 해당 디렉토리에 포함된 파일을 의미함
```
+-- [데이터 전처리]
  - Train 데이터 전처리.ipynb
  - Valid 데이터 전처리.ipynb
  - Test 데이터 전처리.ipynb
+-- [ETRI_dataset]
  +-- [user01-06] # ETRI 나눔에서 다운받은 train 데이터
  +-- [user07-10]
  +-- [user11-12]
  +-- [user21-25]
  +-- [user26-30]
  +-- [train]
    +--[mAcc] : 유저별 Accelerator를 전처리한 데이터
    - acc_final.parquet
    - gps_final.parquet
    - act_final.parquet
    - hr_final.parquet
    - train_valid_date.csv : 유효한 train 날짜 정리한 데이터
  +-- [valid]
    +-- [raw] : 대회 제공 데이터
    +-- [acc] : 유저별 샘플링 데이터, 유저별 필터 및 성분분해 데이터
    +-- [valid_day] : 데이터 전처리 후 하루 단위 모은 데이터
    - acc_final.parquet
    - gps_final.parquet
    - act_final.parquet
    - hr_final.parquet
  +-- [test]
    +-- [raw] : 대회 제공 데이터
    +-- [acc] : 유저별 샘플링 데이터, 유저별 필터 및 성분분해 데이터
    +-- [test_day] : 데이터 전처리 후 하루 단위 모은 데이터
    - acc_final.parquet
    - gps_final.parquet
    - act_final.parquet
    - hr_final.parquet
  - train_label.csv : 대회 제공 라벨 데이터
  - val_label.csv : 대회 제공 라벨 데이터
+-- [log] : 모델 가중치 저장
+-- [result] : csv 파일 저장
cnn_module.py
dataset.py
main.py
test.py
train.py
```


# Dataset
ETRI 나눔 데이터


# How to use
## 1) 
## 2)


# Results
