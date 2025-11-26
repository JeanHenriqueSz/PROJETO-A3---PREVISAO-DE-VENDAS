# ==========================================================
# Aplicação Streamlit para previsão de vendas (Coffee Sales)
# 3 algoritmos: RandomForest, Regressão Linear, MLP Neural
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# CONFIGURAÇÃO DO STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="Previsão de Vendas", layout="wide")
st.title("Previsão de Vendas de Café")
st.write("""
Este sistema permite prever as vendas dos próximos 30 dias
usando três algoritmos de Machine Learning:
RandomForest, Regressão Linear e Rede Neural MLP.
""")


# ---------------------------------------------------------
# PARÂMETROS DO MODELO
# ---------------------------------------------------------
PREVISAO_DIAS = 30
MIN_HISTORICO = 20


# ---------------------------------------------------------
# UPLOAD DOS CSVs (index_1.csv e index_2.csv)
# ---------------------------------------------------------
st.sidebar.header("Upload dos Dados")

arquivo1 = st.sidebar.file_uploader("Envie o arquivo index_1.csv", type=["csv"])
arquivo2 = st.sidebar.file_uploader("Envie o arquivo index_2.csv", type=["csv"])


# ---------------------------------------------------------
# Função de preparação dos dados
# ---------------------------------------------------------
def carregar_e_preparar_dados(file1, file2):

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Junta os dois datasets
    df = pd.concat([df1, df2], ignore_index=True)

    # Normaliza colunas
    df.columns = df.columns.str.lower().str.strip()

    # Colunas essenciais
    if "date" not in df.columns or "coffee_name" not in df.columns:
        raise ValueError("As colunas 'date' e 'coffee_name' são obrigatórias.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "coffee_name"])
    df["coffee_name"] = df["coffee_name"].astype(str).strip()
    df["qtd"] = 1

    # Agregação diária
    vendas_diarias = (
        df.groupby(["date", "coffee_name"], as_index=False)["qtd"]
          .sum()
          .rename(columns={"date": "data", "coffee_name": "produto"})
    )

    vendas_diarias = vendas_diarias.sort_values(["produto", "data"]).reset_index(drop=True)
    return df, vendas_diarias


# ---------------------------------------------------------
# Se arquivos não enviados → parar app
# ---------------------------------------------------------
if arquivo1 is None or arquivo2 is None:
    st.info("Envie os arquivos CSV na barra lateral para continuar.")
    st.stop()


# ---------------------------------------------------------
# Carregar dados
# ---------------------------------------------------------
df_bruto, vendas_diarias = carregar_e_preparar_dados(arquivo1, arquivo2)


st.subheader("Resumo dos Dados")
st.write("Período:", vendas_diarias["data"].min().date(), "→", vendas_diarias["data"].max().date())
st.write("Quantidade de produtos:", vendas_diarias["produto"].nunique())
st.dataframe(vendas_diarias.head())


# ---------------------------------------------------------
# TREINAMENTO DOS MODELOS POR PRODUTO
# ---------------------------------------------------------
st.subheader("Treinamento dos Modelos")

resultados = []

exemplo_produto = None
exemplo_datas = None
exemplo_real = None
exemplo_pred_rf = None
exemplo_pred_lr = None
exemplo_pred_mlp = None

for produto in vendas_diarias["produto"].unique():

    dados = vendas_diarias[vendas_diarias["produto"] == produto].copy()

    # Features
    dados["mes"] = dados["data"].dt.month
    dados["dia"] = dados["data"].dt.day
    dados["dia_semana"] = dados["data"].dt.dayofweek
    dados["lag_1"] = dados["qtd"].shift(1)
    dados["lag_7"] = dados["qtd"].shift(7)

    dados = dados.dropna()

    if len(dados) < MIN_HISTORICO:
        continue

    X = dados[["mes", "dia", "dia_semana", "lag_1", "lag_7"]]
    y = dados["qtd"]

    split = int(len(dados) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if len(X_test) == 0:
        continue

    # -------- Random Forest --------
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    # -------- Linear Regression --------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # -------- MLP Neural Network --------
    mlp = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    pred_mlp = mlp.predict(X_test)

    # Métricas
    mae_rf = mean_absolute_error(y_test, pred_rf)
    mae_lr = mean_absolute_error(y_test, pred_lr)
    mae_mlp = mean_absolute_error(y_test, pred_mlp)

    rmse_rf = mean_squared_error(y_test, pred_rf, squared=False)
    rmse_lr = mean_squared_error(y_test, pred_lr, squared=False)
    rmse_mlp = mean_squared_error(y_test, pred_mlp, squared=False)

    media_diaria = float(np.mean(pred_rf))
    previsao_30d = int(round(media_diaria * PREVISAO_DIAS))

    resultados.append([
        produto, mae_rf, rmse_rf, mae_lr, rmse_lr, mae_mlp, rmse_mlp, media_diaria, previsao_30d
    ])

    if exemplo_produto is None:
        exemplo_produto = produto
        exemplo_datas = dados["data"].iloc[split:]
        exemplo_real = y_test
        exemplo_pred_rf = pred_rf
        exemplo_pred_lr = pred_lr
        exemplo_pred_mlp = pred_mlp


# ---------------------------------------------------------
# RESULTADOS EM TABELA
# ---------------------------------------------------------
cols = [
    "Produto", "MAE_RF", "RMSE_RF", "MAE_LR", "RMSE_LR",
    "MAE_MLP", "RMSE_MLP", "Media diária", "Previsão 30 dias"
]

df_result = pd.DataFrame(resultados, columns=cols)
df_result = df_result.sort_values("Previsão 30 dias", ascending=False)

st.subheader("Resultados por Produto")
st.dataframe(df_result)


# ---------------------------------------------------------
# GRÁFICO TOP 10
# ---------------------------------------------------------
st.subheader("Top 10 produtos mais vendidos (previsão 30 dias)")

top = df_result.head(10)

fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.bar(top["Produto"], top["Previsão 30 dias"])
plt.xticks(rotation=40)
st.pyplot(fig1)


# ---------------------------------------------------------
# GRÁFICO REAL VS PREVISTO
# ---------------------------------------------------------
if exemplo_produto is not None:
    st.subheader("Gráfico Real vs Previsto (Exemplo)")
    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.plot(exemplo_datas, exemplo_real.values, label="Real", marker="o")
    ax2.plot(exemplo_datas, exemplo_pred_rf, label="Random Forest")
    ax2.plot(exemplo_datas, exemplo_pred_lr, label="Linear Regression")
    ax2.plot(exemplo_datas, exemplo_pred_mlp, label="MLP Neural")

    plt.xticks(rotation=40)
    plt.legend()
    st.pyplot(fig2)
