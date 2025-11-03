import pandas as pd
import numpy as np
from faker import Faker
from datetime import date, timedelta
import random
import os

# 1. Configuração Inicial
# ---
# Cria a instância do Faker (para dados em português do Brasil)
fake = Faker('pt_BR')
# Define o período de 6 meses
data_fim = date.today() - timedelta(days=1) # Até ontem
data_inicio = data_fim - timedelta(days=180) # 6 meses (aprox. 180 dias)

# Lista de colunas solicitadas
COLUNAS = [
    'data', 'produto', 'categoria', 'quantidade_vendida',
    'preco_unitario', 'estoque_atual', 'estoque_minimo',
    'dia_promocao', 'fornecedor'
]

# 2. Dados de Base para o Supermercado
# ---
categorias = ['Hortifruti', 'Padaria', 'Laticínios', 'Limpeza', 'Bebidas', 'Mercearia', 'Congelados', 'Carnes']

produtos_por_categoria = {
    'Hortifruti': ['Banana', 'Maçã', 'Alface', 'Tomate', 'Cenoura'],
    'Padaria': ['Pão Francês', 'Bolo de Fubá', 'Pão de Queijo', 'Torta Doce'],
    'Laticínios': ['Leite Integral', 'Queijo Muçarela', 'Iogurte Natural', 'Manteiga'],
    'Limpeza': ['Detergente', 'Água Sanitária', 'Sabão em Pó', 'Desinfetante'],
    'Bebidas': ['Água Mineral', 'Refrigerante Cola', 'Suco de Laranja', 'Cerveja Pilsen'],
    'Mercearia': ['Arroz 5kg', 'Feijão Carioca', 'Óleo de Soja', 'Café em Pó', 'Açúcar Refinado'],
    'Congelados': ['Pizza Calabresa', 'Batata Frita', 'Lasanha Bolonhesa'],
    'Carnes': ['Picanha', 'Frango Filet', 'Linguiça Suína']
}

fornecedores = ['Distribuidora Alfa', 'Produtor Beta', 'Laticínios Gama', 'Limpeza Delta', 'Bebidas Épsilon']

# 3. Geração dos Dados (Linha por Linha)
# ---
dados = []
dias = pd.date_range(start=data_inicio, end=data_fim, freq='D')

# A cada dia, simula um número aleatório de transações (ex: 20 a 50 transações/dia)
for dia in dias:
    num_transacoes = random.randint(20, 50)
    for _ in range(num_transacoes):
        # Seleciona uma categoria e um produto
        categoria = random.choice(categorias)
        produto = random.choice(produtos_por_categoria[categoria])

        # Define características com base na categoria/produto para maior realismo
        if categoria in ['Hortifruti', 'Padaria']:
            # Itens frescos, mais unidades vendidas, preço menor
            preco = round(random.uniform(2.5, 12.0), 2)
            quantidade = random.randint(1, 15)
            estoque_base = random.randint(30, 100)
            estoque_minimo_base = 10
        elif categoria in ['Carnes', 'Bebidas']:
            # Itens mais caros ou com menos unidades por transação
            preco = round(random.uniform(10.0, 50.0), 2)
            quantidade = random.randint(1, 5)
            estoque_base = random.randint(50, 150)
            estoque_minimo_base = 20
        else:
            # Itens gerais
            preco = round(random.uniform(5.0, 30.0), 2)
            quantidade = random.randint(1, 10)
            estoque_base = random.randint(80, 200)
            estoque_minimo_base = 30

        # Simula o estoque atual e mínimo (baseado na venda do dia)
        estoque_atual = estoque_base - quantidade # Simples subtração
        estoque_minimo = random.randint(estoque_minimo_base, estoque_minimo_base + 15)

        # Define se é dia de promoção (ex: 20% dos dias)
        dia_promocao = random.choice([True, False, False, False, False])

        # Monta a linha de dados
        dados.append([
            dia.strftime('%Y-%m-%d'),
            produto,
            categoria,
            quantidade,
            preco,
            estoque_atual,
            estoque_minimo,
            dia_promocao,
            random.choice(fornecedores)
        ])

# 4. Criação do DataFrame e Salvamento no CSV
# ---
df = pd.DataFrame(dados, columns=COLUNAS)

# Garante que o estoque não seja negativo (pode ocorrer com a simulação simples)
df['estoque_atual'] = np.maximum(df['estoque_atual'], 0)

# Nome do arquivo de saída
NOME_ARQUIVO = 'dados_supermercado_simulados.csv'

# Salva o DataFrame como CSV
df.to_csv(NOME_ARQUIVO, index=False, decimal=',', sep=',')

print(f"✅ Arquivo CSV gerado com sucesso!")
print(f"Nome do arquivo: {os.path.abspath(NOME_ARQUIVO)}")
print(f"Total de linhas (transações): {len(df)}")
print(f"Período simulado: {data_inicio} a {data_fim}")