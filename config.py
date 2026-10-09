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
ARQUIVO_COMPARACOES = PASTA_RESULTADOS / "comparacoes.csv"

# ---------------------------------------------------------------------------
# Ações e datas
# ---------------------------------------------------------------------------

# No Yahoo Finance, ações da B3 têm ".SA" no final (o código adiciona sozinho).
TICKERS = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "ABEV3", "WEGE3", "B3SA3",
    "RENT3", "SUZB3", "PRIO3", "RADL3", "EQTL3", "GGBR4", "LREN3", "VIVT3",
    "BOVA11",
]

# Nome das empresas, usado no estado em texto enviado ao Jev e ao LLM (src/estado.py).
NOMES_EMPRESAS = {
    "PETR4": "Petrobras", "VALE3": "Vale", "ITUB4": "Itaú Unibanco", "BBDC4": "Bradesco",
    "BBAS3": "Banco do Brasil", "ABEV3": "Ambev", "WEGE3": "WEG", "B3SA3": "B3",
    "RENT3": "Localiza", "SUZB3": "Suzano", "PRIO3": "PRIO", "RADL3": "Raia Drogasil",
    "EQTL3": "Equatorial", "GGBR4": "Gerdau", "LREN3": "Lojas Renner", "VIVT3": "Vivo",
    "BOVA11": "ETF do Ibovespa",
}

# BOVA11 é o ETF do Ibovespa: a régua de "comprar e segurar".
# Ele é baixado, mas NÃO é uma ação que os modelos podem escolher.
REFERENCIA = "BOVA11"

# Começo do histórico. O ML precisa de passado para treinar.
DATA_INICIO = "2024-01-01"

# Lançamento do Jev (15/09/2026). Pregões a partir daqui com certeza não estavam
# no treino dele: é a "janela limpa" do teste.
DATA_CORTE_JEV = "2026-09-15"

# Avaliação EXTRA, só do ML, num período longo (fora da comparação oficial).
# Serve para saber se a acurácia do ML é a real dele ou ruído de poucos pregões.
# Jev e LLM não entram aqui: esse período pode estar no treino deles.
ML_HISTORICO_INICIO = "2025-01-01"
ML_HISTORICO_FIM = "2025-12-31"
PASTA_EXTRA_ML = RAIZ / "resultados" / "extra" / "ml_2025"

# ---------------------------------------------------------------------------
# Regras de decisão e de avaliação
# ---------------------------------------------------------------------------

# Compra quando a probabilidade de subir passa deste limiar.
LIMIAR_COMPRA = 0.5

# Custo por lado de cada operação, como fração do valor (0,0005 = 0,05%).
CUSTO_POR_LADO = 0.0005

# ---------------------------------------------------------------------------
# Plano de análise (detalhes e justificativas no PREREGISTRO.md)
# Não mude estes valores depois que Jev e LLM começarem a rodar.
# ---------------------------------------------------------------------------

# A análise final só acontece quando a janela limpa tiver este número de pregões
# com resultado (~1 ano a partir de 15/09/2026). Antes disso, tudo é "parcial".
PREGOES_MINIMOS = 250

# O modelo de controle (o ML, congelado como está).
MODELO_CONTROLE = "ml"

# Comparações de Brier (A, B). As PRINCIPAIS respondem à pergunta do projeto e usam
# intervalo de 97,5% (correção de Bonferroni: são 2 testes principais).
# As SECUNDÁRIAS usam 95% e são só descritivas. Todo modelo também é comparado
# com a taxa histórica de altas ("taxa_historica"), como comparação secundária.
COMPARACOES_PRINCIPAIS = [("jev", "ml"), ("jev", "llm")]
COMPARACOES_SECUNDARIAS = [("llm", "ml")]
CONFIANCA_PRINCIPAL = 0.975
CONFIANCA_SECUNDARIA = 0.95

# Baselines que não dão probabilidade de verdade (ficam fora das comparações de Brier).
MODELOS_SEM_PROBABILIDADE = ["sempre_comprar"]

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

# API oficial da TypeSafe (https://docs.typesafe.ai/api.md, consultada em 09/10/2026).
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_TIMEOUT_S = 60
JEV_TENTATIVAS_REDE = 3

# Preço publicado em https://docs.typesafe.ai/models.md (09/10/2026): US$ 0,042 por milhão de
# tokens de ENTRADA (a saída é grátis). A API não devolve o custo, então ele é calculado.
JEV_PRECO_USD_POR_TOKEN = 0.042 / 1_000_000

# O modelo do LLM vem do .env (LLM_MODELO). Temperatura 0 = sempre a resposta mais provável.
LLM_TEMPERATURA = 0

# API compatível com OpenRouter (https://openrouter.ai/docs/quickstart).
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_TIMEOUT_S = 120          # espera máxima por resposta
LLM_TENTATIVAS_REDE = 3      # novas tentativas em erro de rede/servidor (não é a "nova tentativa" por resposta inválida)

# Modelo escolhido (09/10/2026): qwen/qwen3-235b-a22b-2507 (pesos abertos, sem raciocínio,
# lançado em 21/07/2025, treino até 30/06/2025). O nome em si fica no .env (LLM_MODELO).

# Provedor FIXO. Modelos de pesos abertos são hospedados por vários provedores, com
# compressões diferentes (fp4, fp8...) que podem mudar as respostas. Usamos a Alibaba,
# criadora do Qwen. Sem troca automática: se ela cair, a chamada falha em vez de ir para outro lugar.
# Docs: https://openrouter.ai/docs/features/provider-routing
LLM_PROVEDOR = {
    "order": ["alibaba"],
    "allow_fallbacks": False,
    "require_parameters": True,   # recusa provedor que ignoraria temperature/seed
}

# Raciocínio: None = não enviar o parâmetro (o Qwen3 2507 "instruct" não raciocina).
# Para um modelo que raciocina, fixe o nível, por exemplo {"effort": "high"}.
# Docs: https://openrouter.ai/docs/use-cases/reasoning-tokens
LLM_RACIOCINIO = None

# Onde ficam o pré-registro e o registro de falhas das APIs.
ARQUIVO_PREREGISTRO = RAIZ / "PREREGISTRO.md"
ARQUIVO_FALHAS_LLM = PASTA_RESULTADOS / "falhas_llm.csv"   # versionado: transparência sobre o que falhou
ARQUIVO_FALHAS_JEV = PASTA_RESULTADOS / "falhas_jev.csv"
