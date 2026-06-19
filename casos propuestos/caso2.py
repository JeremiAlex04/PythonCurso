import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.datasets import make_classification   # ← Esta era la línea que faltaba

# CASO 2: Predicción de Churn de Clientes

# Creamos un dataset sintético similar al del caso real
X, y = make_classification(
    n_samples=500000,
    n_features=8,
    weights=[0.968, 0.032],   # 3.2% churn
    random_state=42
)

print(f"Dataset cargado: {X.shape[0]:,} clientes | Churn: {y.sum():,}\n")

# División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Modelo Gradient Boosting (como en el material del curso)
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Predicciones y probabilidades
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Resultados
print("=== RESULTADOS DEL MODELO ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Feature Importance (Top 5)
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
importances = pd.Series(model.feature_importances_, index=feature_names)
print("\nTop 5 drivers de Churn:")
print(importances.sort_values(ascending=False).head(5))