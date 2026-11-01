import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# 1. Carregamento dos dados pré-processados da etapa anterior
df = pd.read_csv('steam_games_ai_preprocessed.csv')

# CORREÇÃO DE MEMÓRIA: Pegar uma amostra aleatória representativa de 15.000 linhas
# (Se ainda der erro de memória, diminua o 'n' para 10000. Se rodar com folga, pode tentar 20000)
df = df.sample(n=15000, random_state=42).reset_index(drop=True)

# 2. Separação: O modelo não pode "ver" a variável alvo durante o agrupamento
X = df.drop(columns=['target_class'])
y_true = df['target_class']

# 3. Aplicação do Algoritmo 1: K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans.fit_predict(X)

# 4. Aplicação do Algoritmo 2: Agglomerative Clustering (Hierárquico)
agglo = AgglomerativeClustering(n_clusters=2)
df['cluster_agglo'] = agglo.fit_predict(X)

# 5. Avaliação Métrica: Adjusted Rand Index (ARI)
ari_kmeans = adjusted_rand_score(y_true, df['cluster_kmeans'])
ari_agglo = adjusted_rand_score(y_true, df['cluster_agglo'])

print("=== Métrica de Similaridade com a Variável Alvo ===")
print(f"Adjusted Rand Index (K-Means): {ari_kmeans:.4f}")
print(f"Adjusted Rand Index (Agglomerative): {ari_agglo:.4f}\n")

# 6. Tabelas de Contingência (Cruzamento de Clusters vs. Classe Real)
print("=== Comparativo: K-Means vs Rotulação Existente ===")
crosstab_kmeans = pd.crosstab(
    y_true, df['cluster_kmeans'],
    rownames=['Real (Target Class)'],
    colnames=['Cluster Descoberto (K-Means)']
)
print(crosstab_kmeans)

print("\n=== Comparativo: Agglomerative vs Rotulação Existente ===")
crosstab_agglo = pd.crosstab(
    y_true, df['cluster_agglo'],
    rownames=['Real (Target Class)'],
    colnames=['Cluster Descoberto (Agglo)']
)
print(crosstab_agglo)

# Exportando a base agora com os clusters assinalados para estudos futuros
df.to_csv('steam_games_ai_clustered.csv', index=False)