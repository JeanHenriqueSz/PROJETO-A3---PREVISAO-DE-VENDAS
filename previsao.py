import os                      # para trabalhar com caminhos de arquivos
import pandas as pd            # para manipulação de dados em tabelas (DataFrame)
import numpy as np             # para operações numéricas
from sklearn.ensemble import RandomForestRegressor  # modelo de Machine Learning
import matplotlib.pyplot as plt # para gerar gráficos


# ============================================================
# 1) CONFIGURAÇÕES BÁSICAS
# ============================================================

# Pasta onde estão os arquivos index_1.csv e index_2.csv
BASE_DIR = r"C:\Users\Jeanh\Desktop\testes\PROJETO-A3---PREVISAO-DE-VENDAS\DADOS"

# Caminhos completos dos arquivos de entrada
ARQUIVO1 = os.path.join(BASE_DIR, "index_1.csv")
ARQUIVO2 = os.path.join(BASE_DIR, "index_2.csv")

# Quantos dias queremos prever (próximo mês)
PREVISAO_DIAS = 30

# Nome do arquivo de saída com a previsão final
ARQUIVO_SAIDA = "previsao_cafe_30dias.csv"

# Número mínimo de registros por produto para treinar um modelo decente
MIN_HISTORICO = 20


print("\n=== PREVISÃO DE VENDAS (PRÓXIMO MÊS) - COFFEE SALES ===\n")


# ============================================================
# 2) LEITURA E UNIÃO DOS ARQUIVOS
# ============================================================

print("Lendo arquivos de dados...")

# Lê o primeiro CSV usando pandas
df1 = pd.read_csv(ARQUIVO1)

# Lê o segundo CSV
df2 = pd.read_csv(ARQUIVO2)

# Concatena (empilha) as duas tabelas linha a linha
df = pd.concat([df1, df2], ignore_index=True)

print("Total de linhas após união:", len(df))


# ============================================================
# 3) LIMPEZA INICIAL E NORMALIZAÇÃO
# ============================================================

print("\nLimpando e padronizando dados...")

# Normaliza os nomes das colunas para minúsculo e sem espaços nas pontas
df.columns = df.columns.str.lower().str.strip()

# Verifica se temos as colunas essenciais
colunas_obrigatorias = ["date", "coffee_name"]
for col in colunas_obrigatorias:
    if col not in df.columns:
        raise Exception(f"Coluna obrigatória não encontrada: '{col}'")

# Converte a coluna de data para tipo datetime (data) do pandas
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove linhas onde a data é inválida ou o nome do café é ausente
df = df.dropna(subset=["date", "coffee_name"])

# Garante que o nome do café é string e remove espaços extras
df["coffee_name"] = df["coffee_name"].astype(str).str.strip()

# Cada linha representa 1 venda, então criamos uma coluna quantidade = 1
df["qtd"] = 1

# Mostra o período coberto pela base
print("Período dos dados:", df["date"].min().date(), "→", df["date"].max().date())
print("Total de tipos de café:", df["coffee_name"].nunique())


# ============================================================
# 4) AGREGAÇÃO DIÁRIA POR TIPO DE CAFÉ
# ============================================================

print("\nAgregando vendas diárias por tipo de café...")

# Agrupa por data e tipo de café somando a quantidade de vendas no dia
vendas_diarias = (
    df.groupby(["date", "coffee_name"], as_index=False)["qtd"]
      .sum()
      .rename(columns={"date": "data", "coffee_name": "produto"})
)

# Ordena por produto e data para facilitar a criação de lags
vendas_diarias = vendas_diarias.sort_values(["produto", "data"]).reset_index(drop=True)

print("Total de linhas após agregação:", len(vendas_diarias))


# ============================================================
# 5) TREINO E TESTE COM RANDOM FOREST (POR PRODUTO)
# ============================================================

print("\nTreinando RandomForest para cada produto...")

# Lista de todos os produtos (tipos de café)
produtos = vendas_diarias["produto"].unique()
print("Número de produtos distintos:", len(produtos))

# Lista para armazenar o resultado final de cada produto
resultados = []

# Variáveis para guardar um exemplo de produto para o gráfico real vs previsto
produto_exemplo = None
datas_exemplo = None
y_teste_exemplo = None
y_pred_exemplo = None

for produto in produtos:
    # Seleciona os dados somente daquele produto
    dados = vendas_diarias[vendas_diarias["produto"] == produto].copy()

    # Cria features de tempo a partir da data
    dados["mes"] = dados["data"].dt.month       # mês da venda
    dados["dia"] = dados["data"].dt.day        # dia do mês
    dados["dia_semana"] = dados["data"].dt.dayofweek  # 0=segunda ... 6=domingo

    # Cria lags (memória do histórico de vendas)
    dados["lag_1"] = dados["qtd"].shift(1)     # quantidade vendida no dia anterior
    dados["lag_7"] = dados["qtd"].shift(7)     # quantidade vendida 7 dias atrás

    # Remove as primeiras linhas onde lag_1 ou lag_7 ainda são NaN (sem histórico)
    dados = dados.dropna(subset=["lag_1", "lag_7"])

    # Se após isso o produto tiver poucos registros, pula para não ter modelo ruim
    if len(dados) < MIN_HISTORICO:
        continue

    # ============================
    # 5.1) Definição de X e y
    # ============================

    # X = conjunto de entrada do modelo (features)
    X = dados[["mes", "dia", "dia_semana", "lag_1", "lag_7"]]

    # y = variável alvo (quantidade vendida)
    y = dados["qtd"]

    # ============================
    # 5.2) Separação TREINO x TESTE
    # ============================

    # Ponto de corte: 80% para treino, 20% para teste (sem embaralhar tempo)
    split_index = int(len(dados) * 0.8)

    # Parte de TREINO (primeiros 80% do histórico)
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    # Parte de TESTE (últimos 20% simulando "futuro")
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    # Segurança: se não houver amostras de teste, pula
    if len(X_test) == 0:
        continue

    # ============================
    # 5.3) Treinamento do modelo
    # ============================

    # Cria o modelo RandomForestRegressor com 200 árvores
    modelo = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1  # usa todos os núcleos disponíveis
    )

    # Treina o modelo aprendendo a relação entre X_train e y_train
    modelo.fit(X_train, y_train)

    # ============================
    # 5.4) Avaliação no conjunto de TESTE
    # ============================

    # Faz previsões para o período de teste
    y_pred = modelo.predict(X_test)

    # Calcula a média diária prevista com base nas previsões de teste
    media_diaria_prevista = float(np.mean(y_pred))

    # Projeta a previsão para 30 dias (próximo mês)
    previsao_30_dias = int(round(media_diaria_prevista * PREVISAO_DIAS, 0))

    # Armazena os resultados deste produto
    resultados.append([
        produto,
        round(media_diaria_prevista, 2),
        previsao_30_dias
    ])

    # Guarda um exemplo (primeiro produto que passar aqui) para gráfico de linha
    if produto_exemplo is None:
        produto_exemplo = produto
        datas_exemplo = dados["data"].iloc[split_index:]
        y_teste_exemplo = y_test
        y_pred_exemplo = y_pred


# ============================================================
# 6) TABELA FINAL COM PREVISÕES POR PRODUTO
# ============================================================

# Cria DataFrame com as colunas definidas na lista de resultados
resultado_df = pd.DataFrame(
    resultados,
    columns=["produto", "media_diaria_prevista", "previsao_30_dias"]
)

# Ordena do maior para o menor em termos de previsão para o mês
resultado_df = resultado_df.sort_values("previsao_30_dias", ascending=False).reset_index(drop=True)

print("\nPrevisão de vendas (próximos 30 dias) por tipo de café:\n")
print(resultado_df)

# Salva o resultado em CSV para uso posterior
resultado_df.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")
print("\nArquivo de saída salvo em:", os.path.abspath(ARQUIVO_SAIDA))


# ============================================================
# 7) GRÁFICO 1: BARRAS COM TOP PRODUTOS
# ============================================================

# Seleciona os 10 produtos com maior previsão de vendas
top_n = 10
top = resultado_df.head(top_n)

plt.figure(figsize=(12, 6))
plt.bar(top["produto"], top["previsao_30_dias"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Vendas previstas nos próximos 30 dias")
plt.title("Top produtos - previsão de vendas para o próximo mês")
plt.tight_layout()
plt.show()


# ============================================================
# 8) GRÁFICO 2: REAL vs PREVISTO PARA UM PRODUTO EXEMPLO
# ============================================================

# Só faz o gráfico se tivermos guardado algum produto de exemplo
if produto_exemplo is not None:
    plt.figure(figsize=(12, 6))

    # Plota os valores reais (y_teste_exemplo) ao longo das datas
    plt.plot(datas_exemplo, y_teste_exemplo.values, label="Real", marker="o")

    # Plota as previsões do modelo para o mesmo período
    plt.plot(datas_exemplo, y_pred_exemplo, label="Previsto", marker="x")

    plt.title(f"Real vs Previsto (período de teste) - Produto: {produto_exemplo}")
    plt.xlabel("Data")
    plt.ylabel("Quantidade de vendas")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("\nNenhum produto teve histórico suficiente para gerar gráfico Real vs Previsto.")
