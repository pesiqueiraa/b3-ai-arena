"""
config.py
Todas as configurações do experimento num lugar só.
Se quiser mudar uma ação, uma data ou um custo, é aqui.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Pastas
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent

PASTA_DADOS = RAIZ / "dados"                      # CSVs baixados (fora do git)
ARQUIVO_PRECOS = PASTA_DADOS / "precos_b3.csv"
ARQUIVO_INDICADORES = PASTA_DADOS / "indicadores_b3.csv"

PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_DECISOES = PASTA_RESULTADOS / "decisoes"    # decisoes_<modelo>.csv (versionados no git)
PASTA_CACHE = PASTA_RESULTADOS / "cache"          # respostas das APIs (fora do git)
ARQUIVO_RESULTADO = PASTA_RESULTADOS / "resultado.csv"

# ---------------------------------------------------------------------------
# Ações e datas
# ---------------------------------------------------------------------------

# No Yahoo Finance, ações da B3 têm ".SA" no final (o código adiciona sozinho).
TICKERS = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "ABEV3", "WEGE3", "B3SA3",
    "RENT3", "SUZB3", "PRIO3", "RADL3", "EQTL3", "GGBR4", "LREN3", "VIVT3",
    "BOVA11",
]

# BOVA11 é o ETF do Ibovespa: a régua de "comprar e segurar".
# Ele é baixado, mas NÃO é uma ação que os modelos podem escolher.
REFERENCIA = "BOVA11"

# Começo do histórico. O ML precisa de passado para treinar.
DATA_INICIO = "2024-01-01"

# Lançamento do Jev (15/09/2026). Pregões a partir daqui com certeza não estavam
# no treino dele: é a "janela limpa" do teste.
DATA_CORTE_JEV = "2026-09-15"

# ---------------------------------------------------------------------------
# Regras de decisão e de avaliação
# ---------------------------------------------------------------------------

# Compra quando a probabilidade de subir passa deste limiar.
LIMIAR_COMPRA = 0.5

# Custo por lado de cada operação, como fração do valor (0,0005 = 0,05%).
CUSTO_POR_LADO = 0.0005

# Quantos agentes aleatórios usar no teste contra a sorte.
N_ALEATORIOS = 1000

# Semente dos sorteios. Cada modelo ganha a sua própria sequência derivada desta,
# então incluir um modelo novo não muda o resultado dos outros.
SEMENTE = 42

# ---------------------------------------------------------------------------
# Modelos de API (usados nas próximas etapas)
# ---------------------------------------------------------------------------

# Versão FIXA do Jev. Não use "jev-latest": ela muda sozinha e estraga a comparação.
JEV_VERSAO = "jev-1.13.0"

# O modelo do LLM vem do .env (LLM_MODELO). Temperatura 0 = sempre a resposta mais provável.
LLM_TEMPERATURA = 0
