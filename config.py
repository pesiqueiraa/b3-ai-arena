"""
config.py
Todas as configurações da linha de produção de sinais num lugar só.
Se quiser mudar uma data, um custo ou um parâmetro, é aqui.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Pastas e arquivos
# ---------------------------------------------------------------------------

PASTA_DADOS = RAIZ / "dados"                         # tudo aqui fica fora do git
PASTA_B3 = PASTA_DADOS / "b3"                        # arquivos COTAHIST baixados
PASTA_PROCESSADOS = PASTA_DADOS / "processados"
ARQUIVO_B3 = PASTA_PROCESSADOS / "b3_acoes.csv.gz"
ARQUIVO_PRECOS = PASTA_PROCESSADOS / "precos_ajustados.csv.gz"
ARQUIVO_CALENDARIO = PASTA_PROCESSADOS / "calendario.csv"

PASTA_RESULTADOS = RAIZ / "resultados"               # vai para o git
PASTA_UNIVERSO = PASTA_RESULTADOS / "universo"
ARQUIVO_UNIVERSO = PASTA_UNIVERSO / "universo_mensal.csv"
ARQUIVO_COBERTURA = PASTA_UNIVERSO / "cobertura.csv"
ARQUIVO_TROCA_DE_NOME = PASTA_UNIVERSO / "tickers_com_troca_de_nome.csv"
PASTA_LABORATORIO = PASTA_RESULTADOS / "laboratorio"
ARQUIVO_TENTATIVAS = PASTA_LABORATORIO / "tentativas.csv"
ARQUIVO_ESCOLHA = PASTA_LABORATORIO / "escolha.json"
ARQUIVO_RELATORIO_LAB = PASTA_LABORATORIO / "relatorio.md"
ARQUIVO_PROVA_FINAL = PASTA_RESULTADOS / "prova_final.json"
ARQUIVO_RELATORIO_PROVA = PASTA_RESULTADOS / "prova_final.md"

# ---------------------------------------------------------------------------
# Datas
# ---------------------------------------------------------------------------

DATA_DOWNLOAD_INICIO = "2015-01-01"   # 120 pregões de histórico + 3 meses de liquidez antes de 2016
LAB_INICIO = "2016-01-01"             # laboratório: onde ajustamos tudo
LAB_FIM = "2024-12-31"
RESERVA_INICIO = "2025-01-01"         # reserva TRANCADA: só a prova final olha daqui em diante

# ---------------------------------------------------------------------------
# Fontes
# ---------------------------------------------------------------------------

URL_COTAHIST = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP"

# ---------------------------------------------------------------------------
# Universo (aproximação do IBrX-100 sem viés de sobrevivência)
# ---------------------------------------------------------------------------

TAMANHO_UNIVERSO = 100
JANELA_LIQUIDEZ = 63                  # pregões usados para medir o volume médio (~3 meses)
ESPECIES_ACEITAS = ("ON", "PN", "PNA", "PNB", "PNC", "PND", "UNT")
REFERENCIA = "BOVA11"                 # ETF do Ibovespa: a régua do alvo

# ---------------------------------------------------------------------------
# Alvo, modelo e sinais
# ---------------------------------------------------------------------------

HORIZONTE = 10                        # "vai superar o Ibovespa nos próximos 10 pregões?"
TIPOS_MODELO = ("logistica", "boosting")
SEMENTE = 42

K_PADRAO = 5                          # quantas compras / vendas por dia, no máximo
P_MIN_PADRAO = 0.55                   # probabilidade mínima para um sinal de compra
GRADE_K = (3, 5, 10)                  # grade FIXA e pequena do laboratório (ver especificação 8.2)
GRADE_P_MIN = (0.50, 0.55, 0.60)

# ---------------------------------------------------------------------------
# Carteira simulada e teste contra a sorte
# ---------------------------------------------------------------------------

CUSTO_POR_LADO = 0.001                # 0,05% de custos + 0,05% de margem para slippage
N_ALEATORIOS = 1000

# ---------------------------------------------------------------------------
# Prova final: critérios definidos ANTES de qualquer resultado
# ---------------------------------------------------------------------------

CRITERIO_EXCESSO_MINIMO = 0.0         # excesso líquido total sobre o BOVA11 precisa ser > isto
CRITERIO_ALEATORIOS = 0.90            # superar pelo menos 90% dos agentes aleatórios
CRITERIO_ACERTO_COMPRAS = 0.50        # mais da metade das compras supera o BOVA11
