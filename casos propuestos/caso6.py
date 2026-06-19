# CASO 6: Readmisión Hospitalaria (Gradient Boosting / XGBoost)

import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


X, y = make_classification(n_samples=100000, weights=[0.8, 0.2], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, proba))