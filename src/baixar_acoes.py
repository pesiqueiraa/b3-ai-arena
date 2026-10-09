"""
baixar_acoes.py
Baixa o histórico diário de ações da B3 (via Yahoo Finance) e prepara os dados
para o experimento Jev vs LLM vs ML.

Como rodar no seu computador:
    pip install yfinance pandas
    python baixar_acoes.py

Gera dois arquivos na pasta "dados/":
    precos_b3.csv       -> preços brutos, uma linha por ação por dia
    indicadores_b3.csv  -> indicadores calculados + o gabarito (retorno do dia seguinte)

Dica: rode depois que o pregão fecha (após ~18h) para o dia de hoje vir completo.
"""

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO (mude à vontade)
# ---------------------------------------------------------------------------

# No Yahoo Finance, ações da B3 têm ".SA" no final (o script adiciona sozinho).
# BOVA11 é o ETF do Ibovespa: é a referência de "comprar e segurar".
TICKERS = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "ABEV3", "WEGE3", "B3SA3",
    "RENT3", "SUZB3", "PRIO3", "RADL3", "EQTL3", "GGBR4", "LREN3", "VIVT3",
    "BOVA11",
]

# Começo do histórico. O modelo de ML precisa de passado para treinar,
# então pegamos bem mais do que a janela de teste.
DATA_INICIO = "2024-01-01"

# Lançamento do Jev (15/09/2026). Pregões a partir daqui com certeza não estavam
# no treino dele: é a "janela limpa" do teste.
DATA_CORTE_JEV = "2026-09-15"

PASTA_SAIDA = Path("dados")

COLUNAS_PRECO = ["abertura", "maxima", "minima", "fechamento", "fechamento_ajustado", "volume"]


# ---------------------------------------------------------------------------
# 1) DOWNLOAD
# ---------------------------------------------------------------------------

def baixar_ticker(ticker: str, inicio: str) -> pd.DataFrame:
    """Baixa os preços diários de uma ação e devolve uma tabela com colunas em português."""
    bruto = yf.download(f"{ticker}.SA", start=inicio, auto_adjust=False, progress=False)
    if bruto is None or bruto.empty:
        return pd.DataFrame()

    # Versões recentes do yfinance devolvem colunas em dois níveis,
    # por exemplo ("Close", "PETR4.SA"). Ficamos só com o nome do campo.
    if isinstance(bruto.columns, pd.MultiIndex):
        bruto.columns = bruto.columns.get_level_values(0)
    bruto.columns.name = None

    # "Adj Close" desconta dividendos e desdobramentos. Se não vier, usamos o fechamento.
    if "Adj Close" not in bruto.columns:
        bruto["Adj Close"] = bruto["Close"]

    tabela = bruto.rename(columns={
        "Open": "abertura",
        "High": "maxima",
        "Low": "minima",
        "Close": "fechamento",
        "Adj Close": "fechamento_ajustado",
        "Volume": "volume",
    })[COLUNAS_PRECO]

    tabela.index.name = "data"
    tabela = tabela.reset_index()
    tabela["data"] = pd.to_datetime(tabela["data"])
    if tabela["data"].dt.tz is not None:
        tabela["data"] = tabela["data"].dt.tz_localize(None)
    tabela["data"] = tabela["data"].dt.normalize()

    tabela.insert(1, "ticker", ticker)
    return tabela.dropna(subset=["fechamento"])


# ---------------------------------------------------------------------------
# 2) INDICADORES (o "estado" que os três modelos vão receber)
# ---------------------------------------------------------------------------

def indicadores_de_uma_acao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores que usam SÓ informação disponível até aquele dia.
    Cada conta olha para trás (shift positivo), nunca para frente.
    """
    df = df.sort_values("data").copy()
    preco = df["fechamento_ajustado"]
    volume = df["volume"].replace(0, float("nan"))

    retorno_dia = preco / preco.shift(1) - 1

    df["retorno_1d"] = retorno_dia
    df["retorno_5d"] = preco / preco.shift(5) - 1
    df["retorno_20d"] = preco / preco.shift(20) - 1
    df["distancia_media_20d"] = preco / preco.rolling(20).mean() - 1
    df["volatilidade_20d"] = retorno_dia.rolling(20).std()
    df["volume_relativo_20d"] = volume / volume.rolling(20).mean()

    # GABARITO: quanto a ação rendeu no pregão SEGUINTE.
    # Serve só para dar nota às decisões depois. NUNCA mande esta coluna para o Jev,
    # para o LLM ou como feature do ML: seria entregar a resposta da prova.
    # Na última linha de cada ação ela fica vazia, porque "amanhã" ainda não aconteceu.
    df["gabarito_retorno_amanha"] = preco.shift(-1) / preco - 1
    return df


def calcular_indicadores(precos: pd.DataFrame) -> pd.DataFrame:
    partes = [indicadores_de_uma_acao(df) for _, df in precos.groupby("ticker")]
    ind = pd.concat(partes, ignore_index=True)

    # Marca em qual período cada linha está.
    ind["periodo"] = "historico"
    ind.loc[ind["data"] >= pd.Timestamp(DATA_CORTE_JEV), "periodo"] = "teste_limpo"

    # Os primeiros ~20 pregões de cada ação não têm passado suficiente para os indicadores.
    ind = ind.dropna(subset=["retorno_20d", "volatilidade_20d"])
    return ind.sort_values(["data", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3) PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------

def main() -> None:
    PASTA_SAIDA.mkdir(exist_ok=True)

    tabelas, falharam = [], []
    for ticker in TICKERS:
        print(f"Baixando {ticker:<7}", end=" ", flush=True)
        try:
            tabela = baixar_ticker(ticker, DATA_INICIO)
        except Exception as erro:  # sem internet, ticker mudou de nome, etc.
            print(f"-> erro: {erro}")
            falharam.append(ticker)
            continue
        if tabela.empty:
            print("-> sem dados (ticker existe?)")
            falharam.append(ticker)
            continue
        print(f"-> {len(tabela)} pregões")
        tabelas.append(tabela)
        time.sleep(0.5)  # pausa curta para o Yahoo não bloquear

    if not tabelas:
        raise SystemExit("Nenhuma ação foi baixada. Confira sua conexão com a internet.")

    precos = pd.concat(tabelas, ignore_index=True).sort_values(["data", "ticker"])
    indicadores = calcular_indicadores(precos)

    arq_precos = PASTA_SAIDA / "precos_b3.csv"
    arq_ind = PASTA_SAIDA / "indicadores_b3.csv"
    precos.to_csv(arq_precos, index=False, date_format="%Y-%m-%d")
    indicadores.to_csv(arq_ind, index=False, date_format="%Y-%m-%d", float_format="%.6f")

    limpo = indicadores[indicadores["periodo"] == "teste_limpo"]
    print("\n===== Resumo =====")
    print(f"Ações baixadas: {precos['ticker'].nunique()} de {len(TICKERS)}")
    print(f"De {precos['data'].min():%d/%m/%Y} até {precos['data'].max():%d/%m/%Y}")
    print(f"Pregões na janela limpa (desde {DATA_CORTE_JEV}): {limpo['data'].nunique()}")
    print(f"Arquivos salvos: {arq_precos}  e  {arq_ind}")
    if falharam:
        print(f"Atenção, não vieram: {', '.join(falharam)}")

    print("\nExemplo (último dia de PETR4):")
    exemplo = indicadores[indicadores["ticker"] == "PETR4"].tail(1).copy()
    if not exemplo.empty:
        exemplo["data"] = exemplo["data"].dt.strftime("%d/%m/%Y")
        print(exemplo.T.to_string(header=False))


if __name__ == "__main__":
    main()