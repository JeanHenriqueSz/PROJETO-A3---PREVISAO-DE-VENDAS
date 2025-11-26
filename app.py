# ======================================================================
#  PREVISÃO DE VENDAS DE CAFÉ — Aplicação Streamlit para Deploy na Web
# ======================================================================
#
#  Este script:
#  - Lê os dois arquivos reais do dataset "Coffee Sales"
#  - Trata diferenças de nomes das colunas automaticamente
#  - Remove problemas de encoding / espaços invisíveis
#  - Agrega vendas por dia e por produto
#  - Treina 3 modelos: RandomForest, Regressão Linear e Rede Neural MLP
#  - Calcula métricas MAE e RMSE
#  - Mostra gráficos interativos e tabela final
#  - Permite download da previsão em CSV
#
# ======================================================================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor      # Modelo clássico 1
from sklearn.linear_model import LinearRegression       # Modelo clássico 2
from sklearn.neural_network import MLPRegressor         # Rede Neural
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ======================================================================
# 1) CONFIGURAÇÃO DA PÁGINA
# ======================================================================

st.set_page_config(page_title="Previsão de Café", layout="wide")

st.title("☕ Previsão de Vendas de Café — Machine Learning")
st.write("""
Este sistema utiliza **dados reais do Coffee Sales Dataset** e aplica  
**Random Forest**, **Regressão Linear** e **Rede Neural MLP**  
para prever as vendas dos próximos **30 dias** por tipo de café.
""")


PREVISAO_DIAS = 30
MIN_HISTORICO = 20


# ======================================================================
# 2) UPLOAD DOS ARQUIVOS
# ======================================================================

st.sidebar.header("1. Upload dos Dados")

arquivo1 = st.sidebar.file_uploader("Arquivo index_1.csv", type=["csv"])
arquivo2 = st.sidebar.file_uploader("Arquivo index_2.csv", type=["csv"])

# Se não enviou ambos, para o app
if not arquivo1 or not arquivo2:
    st.info("Envie **os dois arquivos CSV** para iniciar.")
    st.stop()


# ======================================================================
# 3) FUNÇÃO PARA CARREGAR E PREPARAR DADOS
# ======================================================================

def carregar_e_preparar_dados(file1, file2):
    """
    Lê os dois arquivos CSV, normaliza nomes das colunas,
    identifica automaticamente a coluna contendo o nome do café
    e cria a tabela agregada de vendas diárias por produto.
    """

    # Lê arquivos
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Junta os dois datasets
    df = pd.concat([df1, df2], ignore_index=True)

    # Normaliza nomes de colunas
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

    # ------------------------------------------------------------------
    # DETECTA AUTOMATICAMENTE A COLUNA "coffee_name"
    # ------------------------------------------------------------------
    possiveis_nomes = ["coffee_name", "coffee", "name", "product"]

    coluna_produto = None
    for c in df.columns:
        if any(p in c for p in possiveis_nomes):
            coluna_produto = c
            break

    if coluna_produto is None:
        st.error("❌ Nenhuma coluna correspondente ao nome do café foi encontrada.")
        st.stop()

    # Renomeia para "produto"
    df = df.rename(columns={coluna_produto: "produto"})

    # ------------------------------------------------------------------
    # Normaliza coluna de data
    # ------------------------------------------------------------------
    if "date" not in df.columns:
        st.error("❌ A coluna 'date' não existe no CSV enviado.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "produto"])

    # Cada linha é uma venda → quantidade = 1
    df["qtd"] = 1

    # ------------------------------------------------------------------
    # Agregação: vendas diárias por produto
    # ------------------------------------------------------------------
    vendas = (
        df.groupby(["date", "produto"], as_index=False)["qtd"]
          .sum()
          .rename(columns={"date": "data"})
    )

    vendas = vendas.sort_values(["produto", "data"]).reset_index(drop=True)

    return df, vendas


# ======================================================================
# 4) CARREGA OS DADOS
# ======================================================================

with st.spinner("Carregando e preparando dados..."):
    df_raw, vendas_diarias = carregar_e_preparar_dados(arquivo1, arquivo2)

st.subheader("📂 Visão Geral dos Dados")

col1, col2 = st.columns(2)

with col1:
    st.write("**Período dos dados:**")
    st.write(f"{vendas_diarias['data'].min().date()} → {vendas_diarias['data'].max().date()}")
    st.write("**Produtos únicos:**", vendas_diarias["produto"].nunique())
    st.write("**Total de registros agregados:**", len(vendas_diarias))

with col2:
    st.write("Exemplo das vendas agregadas:")
    st.dataframe(vendas_diarias.head(10))


# ======================================================================
# 5) TREINAR OS MODELOS POR PRODUTO
# ======================================================================

st.subheader("🤖 Treinamento dos Modelos")

resultados = []

# Guardar exemplo para gráfico
ex_prod, ex_data, ex_real = None, None, None
ex_pred_rf, ex_pred_lr, ex_pred_mlp = None, None, None

for produto in vendas_diarias["produto"].unique():

    dados = vendas_diarias[vendas_diarias["produto"] == produto].copy()

    # Features de tempo
    dados["mes"] = dados["data"].dt.month
    dados["dia"] = dados["data"].dt.day
    dados["dia_semana"] = dados["data"].dt.dayofweek

    # Lags
    dados["lag_1"] = dados["qtd"].shift(1)
    dados["lag_7"] = dados["qtd"].shift(7)

    dados = dados.dropna()

    if len(dados) < MIN_HISTORICO:
        continue

    # Entradas e saída
    X = dados[["mes", "dia", "dia_semana", "lag_1", "lag_7"]]
    y = dados["qtd"]

    # Split temporal 80 / 20
    split = int(len(dados) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    # ---------------------------------------
    # Modelos
    # ---------------------------------------
    # 1) Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    # 2) Regressão Linear
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # 3) Rede Neural (MLP)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32),
                       max_iter=400, random_state=42)
    mlp.fit(X_train, y_train)
    pred_mlp = mlp.predict(X_test)

    # ---------------------------------------
    # Métricas
    # ---------------------------------------
    mae_rf = mean_absolute_error(y_test, pred_rf)
    rmse_rf = mean_squared_error(y_test, pred_rf, squared=False)

    mae_lr = mean_absolute_error(y_test, pred_lr)
    rmse_lr = mean_squared_error(y_test, pred_lr, squared=False)

    mae_mlp = mean_absolute_error(y_test, pred_mlp)
    rmse_mlp = mean_squared_error(y_test, pred_mlp, squared=False)

    media_prev_dia = float(np.mean(pred_rf))
    previsao_30 = int(round(media_prev_dia * PREVISAO_DIAS))

    resultados.append([
        produto, mae_rf, rmse_rf, mae_lr, rmse_lr, mae_mlp, rmse_mlp,
        media_prev_dia, previsao_30
    ])

    # guarda exemplo
    if ex_prod is None:
        ex_prod = produto
        ex_data = dados["data"].iloc[split:]
        ex_real = y_test
        ex_pred_rf, ex_pred_lr, ex_pred_mlp = pred_rf, pred_lr, pred_mlp


# ======================================================================
# 6) TABELA FINAL DE RESULTADOS
# ======================================================================

cols = [
    "Produto", "MAE_RF", "RMSE_RF", "MAE_LR", "RMSE_LR",
    "MAE_MLP", "RMSE_MLP", "Média Diária Prevista", "Previsão 30 dias"
]

resultados_df = pd.DataFrame(resultados, columns=cols)

st.subheader("📊 Tabela de Resultados")
st.dataframe(resultados_df)


# ======================================================================
# 7) DOWNLOAD DA PREVISÃO
# ======================================================================

csv = resultados_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Baixar CSV", csv, "previsao_30_dias.csv", "text/csv")


# ======================================================================
# 8) GRÁFICO TOP 10 PRODUTOS
# ======================================================================

st.subheader("🏆 Top 10 cafés com maior previsão")

top10 = resultados_df.sort_values("Previsão 30 dias", ascending=False).head(10)

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(top10["Produto"], top10["Previsão 30 dias"])
ax1.set_xticklabels(top10["Produto"], rotation=45, ha="right")
ax1.set_ylabel("Unidades previstas")
st.pyplot(fig1)


# ======================================================================
# 9) GRÁFICO REAL vs PREVISTO
# ======================================================================

st.subheader("📈 Real vs Previsto — Exemplo")

if ex_prod:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(ex_data, ex_real.values, label="Real", marker="o")
    ax2.plot(ex_data, ex_pred_rf, label="RF", marker="x")
    ax2.plot(ex_data, ex_pred_lr, label="Linear", marker="s")
    ax2.plot(ex_data, ex_pred_mlp, label="MLP", marker="^")
    ax2.legend()
    ax2.set_title(f"Real vs Previsto — {ex_prod}")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)
else:
    st.info("Nenhum produto teve histórico suficiente para gerar gráfico de exemplo.")
