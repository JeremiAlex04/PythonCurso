# CASO 1: Detección de Fraude Transaccional (Random Forest)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset: creditcard.csv (Kaggle) - Usar versión de sklearn o descargar
# Para este ejemplo usamos el dataset de sklearn si está disponible o simulamos
from sklearn.datasets import make_classification

# Simulación de dataset desbalanceado (en producción usar el real)
X, y = make_classification(n_samples=284807, n_features=30, weights=[0.9983, 0.0017],random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Fraude')
plt.show()

