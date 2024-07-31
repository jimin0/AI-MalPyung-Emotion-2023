# 🐯 동물의 왕국: 2023 국립국어원 인공 지능 언어 능력 평가 (감정분석 과제)

## 개요
- **대회명**: 2023 국립국어원 인공 지능 언어 능력 평가
- **팀명**: 동물의 왕국
- **과제**: 감정분석
  - **과제 내용**: 인공지능이 문장을 이해하여 주어진 대상에 대한 화자의 감정을 분석하는 과제
  - **평가 방식**: F1 점수로 전체 평가
- **사용 데이터**: 2023 인공지능의 언어 능력 평가: EA 말뭉치
  - **데이터 유형**: 트위터 데이터
  - **감정 카테고리**: 8가지 감정 (joy, anticipation, trust, surprise, disgust, fear, anger, sadness)
- **대회 기간**: 2023.8.21 ~ 2023.10.20
- **최종 순위**: 리더보드 4위

<br>


## 프로젝트 설명
 본 프로젝트는 2023 모두의 말뭉치 언어 능력평가 대회 정량 평가 기준 4위 팀 입니다. 2023 EA 말뭉치(트위터 데이터)를 기반으로 8가지 감정을 분석하는 모델을 개발하였습니다. 

 학습 데이터를 기반으로 감정 분석 모델을 학습시키고, 검증 및 테스트 데이터를 통해 모델의 성능을 평가하였습니다. 이를 위해 Mean, Max Pooling과 특정 단어 주위의 N-gram을 Attention으로 분석하는 두 가지 방식의 모델을 사용하였습니다.



## 프로젝트 구조
```
${PROJECT}
├── EA/
│   ├── data/
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   └── test.jsonl
│   ├── models/
│   │   ├── EnhancedPoolingModel2.py
│   │   └── RealAttention5.py
│   ├── modules/
│   │   ├── arg_parser.py
│   │   ├── dataset_preprocessor.py
│   │   ├── logger_module.py
│   │   └── utils.py
│   └── results/
│   ├── ensemble.py
│   ├── inference_logits.py
│   ├── inference_logitslora.py
│   ├── run.py
│   └── runllm.py
├── Ensemble/
│   └── 가젤왕.jsonl
├── inf.sh
└── requirements.yaml
```

<br>

## 설치 및 실행 방법

### 환경 설정
1. 저장소 클론:
   ```
   git clone https://github.com/jimin0/AI-MalPyung-Emotion-2023.git
   ```

2. 가상 환경 설정:
  -  `requirements.yaml` 파일을 이용하여 가상 환경을 설정합니다.
  - ```conda env create -f requirements.yaml``` 명령어로 가상 환경을 생성합니다.

### 데이터 준비
1. `EA/data/` 폴더에 `train.jsonl`, `dev.jsonl`, `test.jsonl` 파일을 위치시킵니다.
2. `Ensemble/` 폴더에 `가젤왕.jsonl` 파일을 위치시킵니다.

### 모델 실행
프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 추론을 시작합니다.
```
bash inf.sh
```

### 결과 확인
추론 결과는 `Ensemble/최종제출.jsonl` 파일로 저장됩니다.



## 성능
- 감정 분석 (평균) multi_label_classification_micro_F1: `90.1516253`


<br>

## 대회 링크
- [국립국어원 인공지능 (AI)말평](https://kli.korean.go.kr/benchmark)
- [2023 국립국어원 인공 지능 언어 능력 평가: AI 말평 대회 공고문](https://kli.korean.go.kr/benchmark/taskBoardsOrdtm/boardsOrdtm/noticeView.do?page=1&recordId=44&boardOrdtmId=&base.condition=boardOrdtm.title&base.keyword=&size=10)
- [2023 국립국어원 인공지능 언어능력 평가 대회 리더보드](https://kli.korean.go.kr/benchmark/taskOrdtm/taskLeaderBoard.do?taskOrdtmId=103&clCd=END_TASK&subMenuId=sub04)
