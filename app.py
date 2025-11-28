# ======================================================================
#  PREVISÃO DE VENDAS
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

st.title("Previsão de Vendas — Machine Learning")
st.write("""
Este sistema utiliza **dados reais do Coffee Sales Dataset** e aplica  
**Random Forest**, **Regressão Linear** e **Rede Neural MLP**  
para prever as vendas dos próximos **30 dias** por tipo de café.
""")

st.markdown("""
<div style="
    border: 2px solid #4CAF50;
    padding: 15px;
    border-radius: 10px;
    background-color: rgba(76, 175, 80, 0.1);
    text-align: center;
    font-size: 18px;
">
<b>Equipe de Desenvolvimento</b><br><br>

Aparício Virginio do Amaral — 42414535 <br>
Arthur Maestro da Silva Aguiar — 4231925821 <br>
Danielly Silva Teixeira — 42415112 <br>
Jean Henrique Resende Paiva — 4251925073 <br>
Laura Fagundes Freitas — 42413265 <br>
Matheus Henrique Santos — 42410613 <br>
Samuel Fellipe Batista — 42521948 <br>

</div>
""", unsafe_allow_html=True)


PREVISAO_DIAS = 30
MIN_HISTORICO = 20


# ======================================================================
# 2) UPLOAD DO ARQUIVO ÚNICO
# ======================================================================

st.sidebar.header("1. Upload do Arquivo de Dados")

arquivo = st.sidebar.file_uploader("Envie o arquivo CSV ", type=["csv"])

# Se não enviou o CSV, para o app
if not arquivo:
    st.info("Envie o arquivo CSV para iniciar.")
    st.stop()



# ======================================================================
# 3) FUNÇÃO PARA CARREGAR E PREPARAR DADOS
# ======================================================================

def carregar_e_preparar_dados(file):
    df = pd.read_csv(file)

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

    # Detecta automaticamente a coluna do produto
    possiveis_nomes = [
    "produto", "product", "item", "name",
    "corte", "meat", "cut", "coffee", "coffee_name"
]

    coluna_produto = None
    for c in df.columns:
        if any(p in c for p in possiveis_nomes):
            coluna_produto = c
            break

    if coluna_produto is None:
        st.error("❌ Nenhuma coluna correspondente ao nome do café foi encontrada.")
        st.stop()

    df = df.rename(columns={coluna_produto: "produto"})

    # Aceita 'date' ou 'data'
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "data" in df.columns:
        df["date"] = pd.to_datetime(df["data"], errors="coerce")
    else:
        st.error("❌ O CSV precisa ter a coluna 'date' ou 'data'.")
        st.stop()


    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "produto"])

    # Se já existir a coluna qtd no CSV, mantém; se não, assume 1
    if "qtd" not in df.columns:
        df["qtd"] = 1
    else:
        df["qtd"] = pd.to_numeric(df["qtd"], errors="coerce").fillna(1)


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
    df_raw, vendas_diarias = carregar_e_preparar_dados(arquivo)

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

    # Split temporal 70 / 30
    split = int(len(dados) * 0.7)
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
    rmse_rf = mean_squared_error(y_test, pred_rf)

    mae_lr = mean_absolute_error(y_test, pred_lr)
    rmse_lr = mean_squared_error(y_test, pred_lr)

    mae_mlp = mean_absolute_error(y_test, pred_mlp)
    rmse_mlp = mean_squared_error(y_test, pred_mlp)

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
# 6.1) MÉDIAS GERAIS DOS MODELOS
# ======================================================================

st.subheader("📌 Métricas Médias por Modelo")

mae_rf_med = resultados_df["MAE_RF"].mean()
rmse_rf_med = resultados_df["RMSE_RF"].mean()

mae_lr_med = resultados_df["MAE_LR"].mean()
rmse_lr_med = resultados_df["RMSE_LR"].mean()

mae_mlp_med = resultados_df["MAE_MLP"].mean()
rmse_mlp_med = resultados_df["RMSE_MLP"].mean()

# Exibe em forma de tabela
medias_df = pd.DataFrame({
    "Modelo": ["Random Forest", "Linear Regression", "MLP (Rede Neural)"],
    "MAE Médio": [
        round(mae_rf_med, 3),
        round(mae_lr_med, 3),
        round(mae_mlp_med, 3)
    ],
    "RMSE Médio": [
        round(rmse_rf_med, 3),
        round(rmse_lr_med, 3),
        round(rmse_mlp_med, 3)
    ]
})

st.dataframe(medias_df)


# ======================================================================
# 7) DOWNLOAD DA PREVISÃO
# ======================================================================

csv = resultados_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Baixar CSV", csv, "previsao_30_dias.csv", "text/csv")


# ======================================================================
# 8) GRÁFICO TOP 10 PRODUTOS
# ======================================================================

st.subheader("🏆 Top 8 produtos com maior previsão")

top10 = resultados_df.sort_values("Previsão 30 dias", ascending=False).head(10)

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(top10["Produto"], top10["Previsão 30 dias"])
ax1.set_xticklabels(top10["Produto"], rotation=45, ha="right")
ax1.set_ylabel("Unidades previstas")
st.pyplot(fig1)




# ======================================================================
# 9) GRÁFICO REAL vs PREVISTO — AGORA COM DATAS REAIS NO EIXO X
# ======================================================================

st.subheader("📈 Real vs Previsto — Escolha um Produto")

# Lista de produtos disponíveis
lista_produtos = resultados_df["Produto"].unique()

# Seleção do produto
produto_escolhido = st.selectbox(
    "Selecione o produto para visualizar o gráfico:",
    lista_produtos
)

# Filtrar dados do produto escolhido
dados_prod = vendas_diarias[vendas_diarias["produto"] == produto_escolhido].copy()

# Criar features de tempo e lags
dados_prod["mes"] = dados_prod["data"].dt.month
dados_prod["dia"] = dados_prod["data"].dt.day
dados_prod["dia_semana"] = dados_prod["data"].dt.dayofweek
dados_prod["lag_1"] = dados_prod["qtd"].shift(1)
dados_prod["lag_7"] = dados_prod["qtd"].shift(7)

# Remove linhas sem lag
dados_prod = dados_prod.dropna(subset=["lag_1", "lag_7"])

# Se continuar com poucos dados, evita erro
if len(dados_prod) < MIN_HISTORICO:
    st.warning(f"O produto **{produto_escolhido}** não possui histórico suficiente para gerar o gráfico.")
    st.stop()

# X e y
X = dados_prod[["mes", "dia", "dia_semana", "lag_1", "lag_7"]]
y = dados_prod["qtd"]

# Divisão temporal 70/30
split_index = int(len(dados_prod) * 0.7)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

datas_test = dados_prod["data"].iloc[split_index:]

# Se der erro por split vazio, trata
if len(X_train) == 0 or len(X_test) == 0:
    st.warning(f"Dados insuficientes para o produto **{produto_escolhido}**.")
    st.stop()

# Treino dos modelos
modelo_rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)
pred_rf = modelo_rf.predict(X_test)

modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)
pred_lr = modelo_lr.predict(X_test)

modelo_mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
modelo_mlp.fit(X_train, y_train)
pred_mlp = modelo_mlp.predict(X_test)

# Geração do gráfico
fig3, ax3 = plt.subplots(figsize=(12, 5))

ax3.plot(datas_test, y_test.values, label="Real", marker="o")
ax3.plot(datas_test, pred_rf, label="RF", marker="x")
ax3.plot(datas_test, pred_lr, label="Linear", marker="s")
ax3.plot(datas_test, pred_mlp, label="MLP", marker="^")

ax3.set_title(f"Real vs Previsto — {produto_escolhido}")
ax3.set_ylabel("Vendas Diárias")
ax3.legend()
ax3.grid(True)
plt.xticks(rotation=45)

st.pyplot(fig3)

