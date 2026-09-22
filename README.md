# Cognitive Impairment Prediction

50세 이상 인구를 대상으로 **인지장애 경험 여부를 예측**한 이진 분류 프로젝트입니다.  
False Negative의 비용을 고려해 Accuracy보다 **Recall을 우선 지표**로 두고 모델과 임계값을 비교했습니다.

## Problem

클래스 불균형이 있는 분류 문제에서 단순 Accuracy만 높이는 대신, 인지장애 경험자를 놓치지 않는 방향으로 모델을 평가했습니다.

## Experiments

비교·실험한 주요 접근:

- Logistic Regression
- KNN / SVM / Random Forest
- LightGBM / XGBoost / CatBoost
- MLP
- SMOTE
- Soft Voting / Stacking
- Optuna 기반 하이퍼파라미터 탐색
- Threshold optimization

현재 공개된 `main.py`는 **XGBoost + LightGBM + CatBoost + L1 Logistic Regression**을 base estimator로 두고 Logistic Regression을 meta model로 사용하는 Stacking 예시입니다.

## Result

프로젝트 전체 실험 과정에서 양성 클래스 Recall을 **0.72 → 0.78** 수준으로 개선했습니다.

핵심은 모델 하나를 고르는 것보다:

1. 불균형 처리
2. 모델 비교
3. 앙상블
4. 하이퍼파라미터 탐색
5. 의사결정 임계값 조정

을 순차적으로 적용한 것입니다.

## Repository Structure

```text
.
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run

```bash
git clone https://github.com/ryemso/Dementia-Prediction-Machine-Learning-Stacking.git
cd Dementia-Prediction-Machine-Learning-Stacking
pip install -r requirements.txt
```

데이터는 저장소에 포함하지 않습니다. 아래 경로에 분석용 CSV를 준비한 뒤 실행합니다.

```text
data/dementia_data.csv
```

CSV는 `target` 컬럼과 모델 입력 피처를 포함해야 합니다.

```bash
python main.py
```

## Tech Stack

**Python · Pandas · Scikit-learn · XGBoost · LightGBM · CatBoost · Optuna**

## Notes

이 저장소는 프로젝트 전체 실험 중 재현 가능한 Stacking 예시를 정리한 것입니다.  
프로젝트 전체 모델 비교와 결과 요약은 [AI/ML 포트폴리오](https://kimsportpolio.netlify.app/?ver=ai)에서 확인할 수 있습니다.
