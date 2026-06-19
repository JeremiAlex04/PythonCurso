# CASO 5: Cáncer Mamario (SVM Kernel RBF)

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

param_grid = {'C': [1, 10, 100], 'gamma': ['scale', 0.01, 0.001]}
gs = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train_s, y_train)

print("Mejores parámetros:", gs.best_params_)
print("Accuracy:", gs.score(X_test_s, y_test))