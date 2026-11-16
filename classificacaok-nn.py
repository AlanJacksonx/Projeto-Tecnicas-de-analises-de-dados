import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. CARREGAMENTO E INSPEÇÃO DA BASE PRÉ-PROCESSADA
df = pd.read_csv('steam_games_ai_preprocessed.csv')
X = df.drop(columns=['target_class'])
y = df['target_class']

print(f"Dimensões da matriz de atributos (X): {X.shape}")
print("Distribuição relativa da variável alvo:")
print(y.value_counts(normalize=True).round(4))

# 2. PARTICIONAMENTO ESTRATIFICADO (TREINO/OTIMIZAÇÃO vs. AUDITORIA FINAL)
# Reserva 20% dos dados para auditoria cega após a validação cruzada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 3. OTIMIZAÇÃO OBRIGATÓRIA DE HIPERPARÂMETROS COM GridSearchCV
knn_base = KNeighborsClassifier()

# Grade de hiperparâmetros contemplando k, pesos e métrica
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Validação cruzada estratificada interna de 5 folds para o GridSearchCV
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=knn_base,
    param_grid=param_grid,
    cv=cv_inner,
    scoring='accuracy',
    return_train_score=True,
    n_jobs=-1
)

print("\nExecutando busca exaustiva de hiperparâmetros com GridSearchCV...")
grid_search.fit(X_train, y_train)

# Extração do melhor modelo e dos parâmetros vencedores
print("\n=== RESULTADOS DA OTIMIZAÇÃO (GridSearchCV 5 Folds) ===")
print(f"Melhores Hiperparâmetros: {grid_search.best_params_}")
print(f"Melhor Acurácia Média na Busca Interna: {grid_search.best_score_:.4f}")

# Exibe as 5 melhores configurações testadas para análise de sensibilidade
resultados_cv = pd.DataFrame(grid_search.cv_results_)
cols_view = ['rank_test_score', 'param_n_neighbors', 'param_weights',
             'param_metric', 'mean_train_score', 'mean_test_score']
top_5 = resultados_cv.sort_values('rank_test_score')[cols_view].head()
print("\nTop 5 Configurações no GridSearchCV:\n", top_5.to_string(index=False))

# 4. AVALIAÇÃO OBRIGATÓRIA DE DESEMPENHO COM cross_val_score (10 FOLDS)
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
best_knn = grid_search.best_estimator_

# Avaliação principal e complementar para robustez em classes
scores_acc = cross_val_score(best_knn, X_train, y_train, cv=cv_outer, scoring='accuracy')
scores_f1 = cross_val_score(best_knn, X_train, y_train, cv=cv_outer, scoring='f1')
scores_auc = cross_val_score(best_knn, X_train, y_train, cv=cv_outer, scoring='roc_auc')

print("\n=== AVALIAÇÃO CONSOLIDADA VIA cross_val_score (10 Folds) ===")
print(f"Acurácias por Fold: {np.round(scores_acc, 4)}")
print(f"Acurácia Média: {scores_acc.mean():.4f} (+/- {scores_acc.std() * 2:.4f})")
print(f"F1-Score Médio: {scores_f1.mean():.4f} (+/- {scores_f1.std() * 2:.4f})")
print(f"ROC-AUC Médio: {scores_auc.mean():.4f} (+/- {scores_auc.std() * 2:.4f})")

# 5. AUDITORIA FINAL NO CONJUNTO RESERVADO E EXPORTAÇÃO
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

print("\n=== MATRIZ DE CONFUSÃO (CONJUNTO DE TESTE 20%) ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== RELATÓRIO DE CLASSIFICAÇÃO DETALHADO ===")
print(classification_report(y_test, y_pred, target_names=['Misto/Negativo (0)', 'Positivo (1)']))

resultados_cv.to_csv('resultados_gridsearch_knn.csv', index=False)
