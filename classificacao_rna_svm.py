import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. CARREGAMENTO E INSPEÇÃO DA BASE PRÉ-PROCESSADA
df = pd.read_csv('steam_games_ai_preprocessed.csv')
X = df.drop(columns=['target_class'])
y = df['target_class']

print(f"Dimensões da matriz de atributos (X): {X.shape}")
print("Distribuição relativa da variável alvo:")
print(y.value_counts(normalize=True).round(4))

# 2. PARTICIONAMENTO ESTRATIFICADO (TREINO/OTIMIZAÇÃO vs. AUDITORIA FINAL)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ==============================================================================
# 3. MODELO 1: REDE NEURAL ARTIFICIAL - MULTI-LAYER PERCEPTRON (MLP)
# ==============================================================================
print("\n" + "=" * 60)
print("INICIANDO OTIMIZAÇÃO DA RNA (MLPClassifier) COM GridSearchCV")
print("=" * 60)

mlp_base = MLPClassifier(max_iter=300, early_stopping=True, random_state=42)

param_grid_mlp = {
    'hidden_layer_sizes': [(32,), (64,), (64, 32)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

grid_search_mlp = GridSearchCV(
    estimator=mlp_base,
    param_grid=param_grid_mlp,
    cv=cv_inner,
    scoring='accuracy',
    return_train_score=True,
    n_jobs=-1
)

grid_search_mlp.fit(X_train, y_train)
best_mlp = grid_search_mlp.best_estimator_

print(f"\nMelhores Hiperparâmetros (MLP): {grid_search_mlp.best_params_}")
print(f"Melhor Acurácia Média na Busca Interna (5 Folds): {grid_search_mlp.best_score_:.4f}")

# Avaliação rigorosa do melhor MLP com cross_val_score (10 Folds)
mlp_acc = cross_val_score(best_mlp, X_train, y_train, cv=cv_outer, scoring='accuracy', n_jobs=-1)
mlp_f1 = cross_val_score(best_mlp, X_train, y_train, cv=cv_outer, scoring='f1', n_jobs=-1)
mlp_auc = cross_val_score(best_mlp, X_train, y_train, cv=cv_outer, scoring='roc_auc', n_jobs=-1)

print("\n=== AVALIAÇÃO CONSOLIDADA MLP VIA cross_val_score (10 Folds) ===")
print(f"Acurácias por Fold: {np.round(mlp_acc, 4)}")
print(f"Acurácia Média: {mlp_acc.mean():.4f} (+/- {mlp_acc.std() * 2:.4f})")
print(f"F1-Score Médio: {mlp_f1.mean():.4f} (+/- {mlp_f1.std() * 2:.4f})")
print(f"ROC-AUC Médio:  {mlp_auc.mean():.4f} (+/- {mlp_auc.std() * 2:.4f})")

# Auditoria final do MLP no conjunto de teste (20%)
best_mlp.fit(X_train, y_train)
y_pred_mlp = best_mlp.predict(X_test)

print("\nMatriz de Confusão - MLP (Teste 20%):")
print(confusion_matrix(y_test, y_pred_mlp))
print("\nRelatório de Classificação - MLP (Teste 20%):")
print(classification_report(y_test, y_pred_mlp, target_names=['Misto/Negativo (0)', 'Positivo (1)']))

pd.DataFrame(grid_search_mlp.cv_results_).to_csv('resultados_gridsearch_mlp.csv', index=False)

# ==============================================================================
# 4. MODELO 2: SUPPORT VECTOR MACHINE (SVM)
# ==============================================================================
print("\n" + "=" * 60)
print("INICIANDO OTIMIZAÇÃO DO SVM (SVC) COM GridSearchCV")
print("=" * 60)

# Como o SVC possui complexidade quadrática/cúbica em relação ao número de amostras,
# utilizamos uma subamostra estratificada de 15.000 registros para viabilizar a busca em grade
# e a validação cruzada mantendo fielmente a proporção original das classes.
X_train_svm, _, y_train_svm, _ = train_test_split(
    X_train, y_train, train_size=15000, stratify=y_train, random_state=42
)

svm_base = SVC(probability=True, cache_size=1000, random_state=42)

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search_svm = GridSearchCV(
    estimator=svm_base,
    param_grid=param_grid_svm,
    cv=cv_inner,
    scoring='accuracy',
    return_train_score=True,
    n_jobs=-1
)

grid_search_svm.fit(X_train_svm, y_train_svm)
best_svm = grid_search_svm.best_estimator_

print(f"\nMelhores Hiperparâmetros (SVM): {grid_search_svm.best_params_}")
print(f"Melhor Acurácia Média na Busca Interna (5 Folds): {grid_search_svm.best_score_:.4f}")

# Avaliação rigorosa do melhor SVM com cross_val_score (10 Folds)
svm_acc = cross_val_score(best_svm, X_train_svm, y_train_svm, cv=cv_outer, scoring='accuracy', n_jobs=-1)
svm_f1 = cross_val_score(best_svm, X_train_svm, y_train_svm, cv=cv_outer, scoring='f1', n_jobs=-1)
svm_auc = cross_val_score(best_svm, X_train_svm, y_train_svm, cv=cv_outer, scoring='roc_auc', n_jobs=-1)

print("\n=== AVALIAÇÃO CONSOLIDADA SVM VIA cross_val_score (10 Folds) ===")
print(f"Acurácias por Fold: {np.round(svm_acc, 4)}")
print(f"Acurácia Média: {svm_acc.mean():.4f} (+/- {svm_acc.std() * 2:.4f})")
print(f"F1-Score Médio: {svm_f1.mean():.4f} (+/- {svm_f1.std() * 2:.4f})")
print(f"ROC-AUC Médio:  {svm_auc.mean():.4f} (+/- {svm_auc.std() * 2:.4f})")

# Auditoria final do SVM no conjunto de teste completo (20%)
best_svm.fit(X_train_svm, y_train_svm)
y_pred_svm = best_svm.predict(X_test)

print("\nMatriz de Confusão - SVM (Teste 20%):")
print(confusion_matrix(y_test, y_pred_svm))
print("\nRelatório de Classificação - SVM (Teste 20%):")
print(classification_report(y_test, y_pred_svm, target_names=['Misto/Negativo (0)', 'Positivo (1)']))

pd.DataFrame(grid_search_svm.cv_results_).to_csv('resultados_gridsearch_svm.csv', index=False)