import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, confusion_matrix, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import optuna

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

df = pd.read_csv('./data/dementia_data.csv')
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train_scaled, y_train,
                            scoring=make_scorer(recall_score), cv=3).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

best_xgb_params = study.best_params
best_xgb_params.update({
    "random_state": 42, "use_label_encoder": False, "eval_metric": "logloss"
})

xgb = XGBClassifier(**best_xgb_params)
lgb = LGBMClassifier(random_state=42)
cat = CatBoostClassifier(verbose=0, random_state=42)
lasso = LogisticRegression(penalty='l1', solver='saga', class_weight='balanced', max_iter=1000)

meta_model = LogisticRegression(class_weight='balanced')
stack_model = StackingClassifier(
    estimators=[('xgb', xgb), ('lgb', lgb), ('cat', cat), ('lasso', lasso)],
    final_estimator=meta_model,
    passthrough=True,
    cv=5,
    n_jobs=-1
)

stack_model.fit(X_train_scaled, y_train)
y_pred = stack_model.predict(X_test_scaled)

print("📊 Classification Report")
print(classification_report(y_test, y_pred, digits=4))
print("📈 Recall:", recall_score(y_test, y_pred))
print("🔢 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))