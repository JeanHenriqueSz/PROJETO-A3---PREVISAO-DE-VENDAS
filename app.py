import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor          # Modelo clássico 1
from sklearn.linear_model import LinearRegression           # Modelo clássico 2
from sklearn.neural_network import MLPRegressor            # Rede neural
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# 1) CONFIGURAÇÕES GERAIS DO APP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Previsão de Vendas de Café",
    layout="wide"
)

st.title("☕ Previsão de Vendas de Café com Machine Learning")
st.write(
    """
    Este aplicativo utiliza **dados reais de vendas (Coffee Sales)** e aplica 
    **dois algoritmos clássicos de Machine Learning** e **uma Rede Neural** para
    prever as vendas dos próximos 30 dias por tipo de café.
    """
)

# Parâmetros gerais do experimento
PREVISAO_DIAS = 30          # horizonte de previsão (30 dias)
MIN_HISTORICO = 20          # mínimo de registros por produto para treinar


# ---------------------------------------------------------
# 2) UPLOAD DOS ARQUIVOS CSV (index_1 e index_2)
# ---------------------------------------------------------

st.sidebar.header("1. Upload dos Dados")

st.sidebar.write(
    "Envie os dois arquivos do dataset **Coffee Sales**: "
    "`index_1.csv` e `index_2.csv`."
)

arquivo1 = st.sidebar.file_uploader("Arquivo index_1.csv", type=["csv"])
arquivo2 = st.sidebar.file_uploader("Arquivo index_2.csv", type=["csv"])


# Função auxiliar para ler e preparar o dataset
def carregar_e_preparar_dados(file1, file2):
    """
    Lê os dois arquivos CSV, concatena e faz uma limpeza básica.

    Retorna:
        df (DataFrame): dados brutos tratados
        vendas_diarias (DataFrame): vendas agregadas por dia e produto
    """
    # Lê os CSVs enviados
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Junta os dois datasets linha a linha
    df = pd.concat([df1, df2], ignore_index=True)

    # Padroniza nomes das colunas
    df.columns = df.columns.str.lower().str.strip()

    # Garanta que as colunas essenciais existam
    if "date" not in df.columns or "coffee_name" not in df.columns:
        raise ValueError("As colunas 'date' e 'coffee_name' são obrigatórias na base.")

    # Converte a coluna de data para tipo datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Remove linhas inválidas (data vazia ou nome de café ausente)
    df = df.dropna(subset=["date", "coffee_name"])

    # Normaliza o nome do café
    df["coffee_name"] = df["coffee_name"].astype(str).str.strip()

    # Cada linha é uma venda → criamos uma coluna quantidade = 1
    df["qtd"] = 1

    # Agregação diária: soma de vendas por dia e por produto
    vendas_diarias = (
        df.groupby(["date", "coffee_name"], as_index=False)["qtd"]
          .sum()
          .rename(columns={"date": "data", "coffee_name": "produto"})
    )

    # Ordena por produto e data (importante para lags e split temporal)
    vendas_diarias = vendas_diarias.sort_values(["produto", "data"]).reset_index(drop=True)

    return df, vendas_diarias


# Só prossegue se os dois arquivos forem enviados
if arquivo1 is None or arquivo2 is None:
    st.info("➡ Envie os dois arquivos CSV na barra lateral para começar.")
    st.stop()

# ---------------------------------------------------------
# 3) CARREGAR E MOSTRAR DADOS BÁSICOS
# ---------------------------------------------------------

with st.spinner("Carregando e preparando dados..."):
    df_bruto, vendas_diarias = carregar_e_preparar_dados(arquivo1, arquivo2)

st.subheader("📂 Visão Geral dos Dados")
col1, col2 = st.columns(2)

with col1:
    st.write("**Período dos dados:**")
    st.write(f"{vendas_diarias['data'].min().date()} → {vendas_diarias['data'].max().date()}")
    st.write("**Quantidade total de registros agregados (dia x produto):**", len(vendas_diarias))
    st.write("**Quantidade de tipos de café (produtos):**", vendas_diarias["produto"].nunique())

with col2:
    st.write("**Exemplo de registros agregados (vendas diárias por produto):**")
    st.dataframe(vendas_diarias.head(10))


# ---------------------------------------------------------
# 4) TREINAR MODELOS PARA CADA PRODUTO
# ---------------------------------------------------------

st.subheader("🤖 Treinamento dos Modelos (por produto)")

st.write(
    f"""
    Para cada tipo de café:
    - Criamos features de tempo (mês, dia, dia da semana)  
    - Criamos lags de vendas (`lag_1` e `lag_7`)  
    - Separamos **80% para treino** e **20% para teste** (respeitando a ordem temporal)  
    - Treinamos **3 modelos**:
        - RandomForestRegressor (clássico 1)  
        - LinearRegression (clássico 2)  
        - MLPRegressor (rede neural)  
    - Calculamos **MAE** e **RMSE** no conjunto de teste  
    - Estimamos a média diária prevista e projetamos os próximos 30 dias
    """
)

# DataFrame para armazenar métricas de todos os produtos
resultados_metricas = []

# Lista para guardar alguns dados de exemplo para gráfico Real x Previsto
exemplo_produto = None
exemplo_datas = None
exemplo_real = None
exemplo_pred_rf = None
exemplo_pred_lr = None
exemplo_pred_mlp = None

# Loop em cada produto (tipo de café)
for produto in vendas_diarias["produto"].unique():
    # Filtra apenas as vendas daquele produto
    dados = vendas_diarias[vendas_diarias["produto"] == produto].copy()

    # Criação de features de tempo
    dados["mes"] = dados["data"].dt.month
    dados["dia"] = dados["data"].dt.day
    dados["dia_semana"] = dados["data"].dt.dayofweek

    # Criação dos lags (memória das vendas anteriores)
    dados["lag_1"] = dados["qtd"].shift(1)
    dados["lag_7"] = dados["qtd"].shift(7)

    # Remove linhas iniciais sem lag (NaN)
    dados = dados.dropna(subset=["lag_1", "lag_7"])

    # Se não tiver histórico suficiente, pula o produto
    if len(dados) < MIN_HISTORICO:
        continue

    # -----------------------------
    # 4.1) Definição de X (entradas) e y (saída)
    # -----------------------------
    X = dados[["mes", "dia", "dia_semana", "lag_1", "lag_7"]]
    y = dados["qtd"]

    # -----------------------------
    # 4.2) Separação TREINO (80%) e TESTE (20%)
    # -----------------------------
    split_index = int(len(dados) * 0.8)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    # Segurança: se não houver teste, pula
    if len(X_test) == 0:
        continue

    # -----------------------------
    # 4.3) Treinamento dos 3 modelos
    # -----------------------------

    # Modelo clássico 1: Random Forest
    modelo_rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)

    # Modelo clássico 2: Regressão Linear
    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train, y_train)
    y_pred_lr = modelo_lr.predict(X_test)

    # Modelo 3: Rede Neural MLP
    modelo_mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )
    modelo_mlp.fit(X_train, y_train)
    y_pred_mlp = modelo_mlp.predict(X_test)

    # -----------------------------
    # 4.4) Cálculo das Métricas (MAE e RMSE)
    # -----------------------------
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)

    mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
    rmse_mlp = mean_squared_error(y_test, y_pred_mlp, squared=False)

    # -----------------------------
    # 4.5) Previsão média para 30 dias (usando Random Forest)
    # -----------------------------
    media_diaria_prevista = float(np.mean(y_pred_rf))
    previsao_30_dias = int(round(media_diaria_prevista * PREVISAO_DIAS))

    # Guarda as métricas deste produto
    resultados_metricas.append({
        "produto": produto,
        "MAE_RF": mae_rf,
        "RMSE_RF": rmse_rf,
        "MAE_LR": mae_lr,
        "RMSE_LR": rmse_lr,
        "MAE_MLP": mae_mlp,
        "RMSE_MLP": rmse_mlp,
        "media_diaria_prevista": media_diaria_prevista,
        "previsao_30_dias": previsao_30_dias
    })

    # Guarda um produto de exemplo para gráfico
    if exemplo_produto is None:
        exemplo_produto = produto
        exemplo_datas = dados["data"].iloc[split_index:]
        exemplo_real = y_test
        exemplo_pred_rf = y_pred_rf
        exemplo_pred_lr = y_pred_lr
        exemplo_pred_mlp = y_pred_mlp

# Converte lista de dicionários para DataFrame
if len(resultados_metricas) == 0:
    st.error("Nenhum produto teve histórico suficiente para treinar os modelos.")
    st.stop()

resultados_df = pd.DataFrame(resultados_metricas)


# ---------------------------------------------------------
# 5) VISUALIZAÇÃO DOS RESULTADOS GERAIS
# ---------------------------------------------------------

st.subheader("📊 Resultados Gerais por Modelo")

# Cálculo das médias das métricas (média MAE e RMSE por modelo)
resumo_modelos = pd.DataFrame({
    "Modelo": ["Random Forest", "Regressão Linear", "MLP (Rede Neural)"],
    "MAE_médio": [
        resultados_df["MAE_RF"].mean(),
        resultados_df["MAE_LR"].mean(),
        resultados_df["MAE_MLP"].mean()
    ],
    "RMSE_médio": [
        resultados_df["RMSE_RF"].mean(),
        resultados_df["RMSE_LR"].mean(),
        resultados_df["RMSE_MLP"].mean()
    ]
})

st.write("**Métricas médias (quanto menor, melhor):**")
st.dataframe(resumo_modelos.style.format({"MAE_médio": "{:.3f}", "RMSE_médio": "{:.3f}"}))

# Tabela com previsão de 30 dias por produto (ordenada)
st.write("### Previsão de vendas para os próximos 30 dias (por tipo de café)")
tabela_previsao = resultados_df[["produto", "media_diaria_prevista", "previsao_30_dias"]].copy()
tabela_previsao = tabela_previsao.sort_values("previsao_30_dias", ascending=False).reset_index(drop=True)
tabela_previsao.columns = ["Produto", "Média diária prevista", "Previsão 30 dias"]
st.dataframe(tabela_previsao)

# Botão para download da tabela como CSV
csv_download = tabela_previsao.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Baixar previsões em CSV",
    data=csv_download,
    file_name="previsao_cafe_30_dias.csv",
    mime="text/csv"
)


# ---------------------------------------------------------
# 6) GRÁFICO DE BARRAS COM TOP PRODUTOS
# ---------------------------------------------------------

st.subheader("🏆 Top 10 cafés com maior previsão de vendas (30 dias)")

top_n = 10
top_produtos = tabela_previsao.head(top_n)

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(top_produtos["Produto"], top_produtos["Previsão 30 dias"])
ax1.set_xticklabels(top_produtos["Produto"], rotation=45, ha="right")
ax1.set_ylabel("Unidades previstas (30 dias)")
ax1.set_title("Top 10 produtos previstos para o próximo mês")

st.pyplot(fig1)


# ---------------------------------------------------------
# 7) GRÁFICO REAL vs PREVISTO PARA UM PRODUTO EXEMPLO
# ---------------------------------------------------------

st.subheader("📈 Real vs Previsto – Exemplo de Produto")

if exemplo_produto is not None:
    st.write(f"Produto de exemplo: **{exemplo_produto}** (período de teste)")

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    # Série real
    ax2.plot(exemplo_datas, exemplo_real.values, label="Real", marker="o")

    # Previsões dos 3 modelos
    ax2.plot(exemplo_datas, exemplo_pred_rf, label="Previsto RF", marker="x")
    ax2.plot(exemplo_datas, exemplo_pred_lr, label="Previsto LR", marker="s")
    ax2.plot(exemplo_datas, exemplo_pred_mlp, label="Previsto MLP", marker="^")

    ax2.set_xlabel("Data")
    ax2.set_ylabel("Vendas")
    ax2.set_title(f"Real vs Previsto – Produto: {exemplo_produto}")
    ax2.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    st.pyplot(fig2)
else:
    st.info("Nenhum produto com histórico suficiente para gerar gráfico de exemplo.")
