# CASO 4: Diabetes Tipo 2 (LogReg + Random Forest)


from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_openml(data_id=37, as_frame=True)  # Pima Indians
# La columna 'target' contiene texto ('tested_positive', 'tested_negative').
# Lo convertimos a 1 y 0 para que los modelos puedan usarlo.
X, y = data.data, (data.target == 'tested_positive').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Modelos
logreg = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("LogReg CV AUC:", cross_val_score(logreg, X_train_s, y_train, cv=cv, scoring='roc_auc').mean())
print("RF CV AUC:", cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc').mean())