# CASO 3: Lavado de Activos (AML) - Cascada Isolation Forest + GBM

from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
import numpy as np

# Simulación de un dataset para AML (muy desbalanceado)
# En un caso real, aquí se cargarían los datos.
X, y = make_classification(
    n_samples=100000,
    n_features=20,
    weights=[0.98, 0.02], # 2% de casos sospechosos, coincide con 'contamination'
    random_state=42
)

# Etapa 1: Detección de anomalías (no supervisado)
iso = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
anomalies = iso.fit_predict(X) == -1

X_anom = X[anomalies]
y_anom = y[anomalies] # Filtramos también las etiquetas verdaderas para evaluar el modelo final

# Etapa 2: Clasificación supervisada
X_tr, X_te, y_tr, y_te = train_test_split(X_anom, y_anom, test_size=0.3, random_state=42)

gbm = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
gbm.fit(X_tr, y_tr)

print(classification_report(y_te, gbm.predict(X_te)))