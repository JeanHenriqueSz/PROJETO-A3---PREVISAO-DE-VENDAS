import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# =======================
# 1) Carregar CSV
# =======================
df = pd.read_csv(
    r"D:\UNA\IA-Sexta\PROJETO-A3---PREVISAO-DE-VENDAS\dados_supermercado_simulados.csv",
    sep=None,
    engine='python'
)

# =======================
# 2) Limpar colunas
# =======================
df.columns = df.columns.str.lower().str.replace('"', '').str.strip()

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.replace('"', '').str.strip()

# =======================
# 3) Garantir tipos corretos
# =======================
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values('data')

# =======================
# 4) Previsão produto por produto
# =======================
resultado_previsoes = []

for produto in df['produto'].unique():

    dados = df[df['produto'] == produto].copy()

    dados['mes'] = dados['data'].dt.month
    dados['dia'] = dados['data'].dt.day
    dados['dia_semana'] = dados['data'].dt.dayofweek

    dados['lag_1'] = dados['quantidade_vendida'].shift(1)
    dados['lag_7'] = dados['quantidade_vendida'].shift(7)

    dados = dados.dropna()

    if len(dados) < 14:
        continue

    X = dados[['mes', 'dia', 'dia_semana', 'lag_1', 'lag_7']]
    y = dados['quantidade_vendida']

    split = int(len(dados) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    previsoes = model.predict(X_test)
    media_prevista = np.mean(previsoes)

    resultado_previsoes.append([produto, round(media_prevista, 2)])

# =======================
# 5) Gerar previsão mensal já arredondada
# =======================
resultado_df = pd.DataFrame(resultado_previsoes, columns=["produto", "media_diaria"])
resultado_df["previsao_30dias"] = (resultado_df["media_diaria"] * 30).round().astype(int)

# =======================
# 6) Trazer o estoque atual
# =======================
df_estoque = df.groupby("produto")["estoque_atual"].last().reset_index()
resultado_df = resultado_df.merge(df_estoque, on="produto", how="left")

# Garantir que estoque seja inteiro
resultado_df["estoque_atual"] = resultado_df["estoque_atual"].astype(int)

# =======================
# 7) Calcular quantidade ideal a comprar (sempre arredondando pra cima)
# =======================
resultado_df["comprar"] = resultado_df["previsao_30dias"] - resultado_df["estoque_atual"]
resultado_df["comprar"] = resultado_df["comprar"].apply(lambda x: x if x > 0 else 0)
resultado_df["comprar"] = resultado_df["comprar"].round().astype(int)

# Ordenar
lista = resultado_df[['produto', 'previsao_30dias', 'estoque_atual', 'comprar']]
lista = lista.sort_values(by="comprar", ascending=False)

# =======================
# 8) LISTA DE COMPRAS FORMATADA (VISUAL)
# =======================

# Determinar período analisado
data_inicio = df['data'].min().strftime("%d/%m/%Y")
data_fim = df['data'].max().strftime("%d/%m/%Y")

lista = resultado_df[['produto', 'previsao_30dias', 'estoque_atual', 'comprar']]
lista = lista.sort_values(by="comprar", ascending=False)

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

table = ax.table(
    cellText=lista.values,
    colLabels=["Produto", "Vendas Previstas (30 dias)", "Estoque Atual", "Comprar"],
    loc='center',
    cellLoc='center',
    colLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.3, 1.6)

plt.title(
    f" Lista de Compras Recomendada para o Próximo Mês\n"
    f"Período analisado: {data_inicio} até {data_fim}",
    fontsize=18,
    pad=100
)

plt.tight_layout()
plt.show()


# =======================
# 9) Salvar lista em CSV
# =======================
lista.to_csv("lista_de_compras.csv", index=False)
print("\n💾 Lista salva como lista_de_compras.csv — pronta para imprimir ou enviar.\n")
