import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# 1. Carregamento dos dados pré-processados
df = pd.read_csv('steam_games_ai_preprocessed.csv')

# Separação entre atributos previsores (X) e a classe alvo (y)
X = df.drop(columns=['target_class'])
y = df['target_class']

print("Iniciando treinamento e validação dos classificadores...\n")

# 2. Modelo 1: Naive Bayes (GaussianNB escolhido devido aos dados padronizados)
nb_classifier = GaussianNB()

# Avaliação rigorosa utilizando Validação Cruzada (10 Folds)
scores_nb = cross_val_score(nb_classifier, X, y, cv=10, scoring='accuracy')

print("=== Desempenho do Modelo: Naive Bayes ===")
print(f"Acurácias em cada fold: {scores_nb}")
print(f"Acurácia Média Consolidada: {scores_nb.mean():.4f} (+/- {scores_nb.std() * 2:.4f})")

# 3. Modelo 2: Árvore de Decisão
dt_classifier = DecisionTreeClassifier(random_state=42)

# Definindo a grade (grid) de hiperparâmetros a serem testados
param_grid = {
    'criterion': ['gini', 'entropy'],       # Métricas de qualidade da divisão
    'max_depth': [None, 5, 10, 15, 20],      # Profundidade máxima da árvore
    'min_samples_split': [2, 5, 10],         # Mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4]            # Mínimo de amostras permitidas em uma folha
}

# 4. Busca exaustiva pelos melhores parâmetros
# O GridSearchCV já realiza validação cruzada internamente (cv=5) durante a busca
grid_search = GridSearchCV(
    estimator=dt_classifier, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1 # Usa todos os núcleos do processador para acelerar
)

# Executa o treinamento da grade
grid_search.fit(X, y)

# Resgata o modelo otimizado com a melhor combinação encontrada
best_dt_model = grid_search.best_estimator_

print("\n=== Otimização da Árvore de Decisão (GridSearchCV) ===")
print(f"Melhores Hiperparâmetros Encontrados: {grid_search.best_params_}")

# 5. Avaliação do Melhor Modelo utilizando cross_val_score (10 Folds)
# Exigência metodológica de aplicar cross_val_score em todos os classificadores finais
scores_dt = cross_val_score(best_dt_model, X, y, cv=10, scoring='accuracy')

print("\n=== Desempenho do Modelo: Árvore de Decisão (Otimizada) ===")
print(f"Acurácias em cada fold: {scores_dt}")
print(f"Acurácia Média Consolidada: {scores_dt.mean():.4f} (+/- {scores_dt.std() * 2:.4f})")
