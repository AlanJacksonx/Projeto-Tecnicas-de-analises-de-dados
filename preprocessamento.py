import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Carregamento dos Dados
df = pd.read_csv('base_de_dados_jogos.csv')

# Padroniza os nomes das colunas
df.columns = df.columns.str.strip().str.lower()

print("Formato original:", df.shape)

# 2. Limpeza Inicial e Filtros
# ATENÇÃO: Como não há coluna de data no seu CSV, o filtro de ano (2024) foi removido.

# Remover jogos que não possuem nota ('review_score') 
df.dropna(subset=['review_score'], inplace=True)

# 3. Engenharia de Atributos (Feature Engineering)
# Criando a variável alvo (Target) baseada na nota 'review_score'
df['target_class'] = np.where(df['review_score'] >= 70, 1, 0)

# A coluna 'uses_ai' já existe no seu dataset, então não precisamos criá-la!

# 4. Tratamento de Outliers e Assimetria (Log Transform)
df['log_peak_ccu'] = np.log1p(df['peak_ccu'])
df['log_total_reviews'] = np.log1p(df['total_positive'] + df['total_negative'])

# 5. Separação de Variáveis por Tipo para o Pipeline
# Incluímos o 'uses_ai' direto nos numéricos, já que ele já vem no CSV
numeric_features = ['price', 'log_peak_ccu', 'log_total_reviews', 'uses_ai']

# O gênero no seu arquivo se chama 'genre_tag'
categorical_features = ['genre_tag'] 

# 6. Construção do Pipeline de Pré-processamento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Preenche vazios com a mediana
    ('scaler', StandardScaler())                   # Padroniza (média 0, desvio padrão 1)
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), # Preenche vazios
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Transforma em 0s e 1s
])

# Combinando os processadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 7. Aplicação do Pré-processamento
X = df[numeric_features + categorical_features]
y = df['target_class']

# Treinando e aplicando a transformação
X_preprocessed = preprocessor.fit_transform(X)

# Recuperando o nome das colunas geradas pelo OneHotEncoder
cat_columns = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_columns = numeric_features + list(cat_columns)

# Criando o DataFrame final processado e pronto para o Machine Learning
df_processed = pd.DataFrame(X_preprocessed, columns=all_columns)
df_processed['target_class'] = y.values

print("Formato após pré-processamento:", df_processed.shape)
print(df_processed.head()) 

# Salvando o arquivo limpo para as próximas etapas do trabalho
df_processed.to_csv('steam_games_ai_preprocessed.csv', index=False)