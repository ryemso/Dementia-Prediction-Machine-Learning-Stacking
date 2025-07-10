# 🧠 Dementia Prediction using Stacking Ensemble (with Optuna)

이 프로젝트는 50세 이상 인구를 대상으로 **인지장애 여부**를 예측하는 머신러닝 모델입니다.
다양한 개별 모델을 결합한 **스태킹 앙상블 구조**를 사용하며, 핵심 평가지표는 `Recall`입니다.

## 📌 주요 특징
- XGBoost, LightGBM, CatBoost, Lasso 등의 모델 조합
- Optuna 기반 자동 하이퍼파라미터 튜닝
- LogisticRegression 기반 메타 모델 스태킹
- 성능 평가 지표: Recall, Classification Report

## 🛠 기술 스택
- Python, Pandas, Scikit-learn, XGBoost, CatBoost, Optuna

## 🚀 실행 방법

```bash
git clone https://github.com/your-username/dementia-stacking-model.git
cd dementia-stacking-model
pip install -r requirements.txt
python main.py
```

## 📂 데이터
- `./data/dementia_data.csv` 파일 필요
  - 열 이름: `target` (이진 타깃), 그 외 특성 변수들