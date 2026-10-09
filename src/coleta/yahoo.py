"""
yahoo.py
Preços AJUSTADOS por dividendos e desdobramentos (Yahoo Finance). São usados nos retornos, nas
features e no alvo. No Yahoo, ações da B3 têm ".SA" no final (o código adiciona sozinho).
"""

import time

import pandas as pd

COLUNAS = ["data", "ticker", "fechamento_ajustado", "volume"]


def baixar_ticker(ticker: str, inicio: str) -> pd.DataFrame:
    """Baixa um ticker. Devolve tabela vazia se o Yahoo não tiver dados (ex.: ação que deixou de existir)."""
    import yfinance as yf  # importado aqui para os testes rodarem sem internet

    bruto = yf.download(f"{ticker}.SA", start=inicio, auto_adjust=False, progress=False)
    if bruto is None or bruto.empty:
        return pd.DataFrame()
    if isinstance(bruto.columns, pd.MultiIndex):           # versões novas: ("Close", "PETR4.SA")
        bruto.columns = bruto.columns.get_level_values(0)
    if "Adj Close" not in bruto.columns:
        bruto["Adj Close"] = bruto["Close"]
    tabela = bruto.rename(columns={
        "Open": "abertura", "High": "maxima", "Low": "minima",
        "Adj Close": "fechamento_ajustado", "Volume": "volume",
    })[["abertura", "maxima", "minima", "fechamento_ajustado", "volume"]].copy()
    tabela.index.name = "data"
    tabela = tabela.reset_index()
    tabela["data"] = pd.to_datetime(tabela["data"])
    if tabela["data"].dt.tz is not None:
        tabela["data"] = tabela["data"].dt.tz_localize(None)
    tabela["data"] = tabela["data"].dt.normalize()
    tabela.insert(1, "ticker", ticker)
    return tabela.dropna(subset=["fechamento_ajustado"])


def remover_pregoes_incompletos(precos: pd.DataFrame) -> pd.DataFrame:
    """
    Quando o download acontece antes de o Yahoo consolidar o dia, a linha de hoje vem com abertura,
    máxima e mínima iguais a zero. Essas linhas são removidas. Dias antigos com volume zero mas com
    preços preenchidos (dia sem negócio) ficam.
    """
    incompleta = (precos["abertura"] == 0) & (precos["maxima"] == 0) & (precos["minima"] == 0)
    if incompleta.any():
        datas = sorted(precos.loc[incompleta, "data"].dt.strftime("%d/%m/%Y").unique())
        print(f"Aviso: {int(incompleta.sum())} linha(s) de pregão incompleto removidas "
              f"(datas: {', '.join(datas)}). Baixe de novo depois do fechamento.")
    return precos[~incompleta].copy()


def remover_datas_repetidas(precos: pd.DataFrame) -> pd.DataFrame:
    """Se o Yahoo devolver a mesma data duas vezes para um ticker, fica a última."""
    return precos.drop_duplicates(subset=["data", "ticker"], keep="last").copy()


def baixar_varios(tickers: list[str], inicio: str, pausa: float = 0.3,
                  baixar=baixar_ticker) -> tuple[pd.DataFrame, list[str]]:
    """Baixa vários tickers, limpa e junta. Devolve (precos, tickers_que_falharam)."""
    tabelas, falharam = [], []
    for numero, ticker in enumerate(tickers, start=1):
        print(f"  Yahoo {numero}/{len(tickers)}: {ticker:<8}", end=" ", flush=True)
        try:
            tabela = baixar(ticker, inicio)
        except Exception as erro:  # sem internet, ticker sem dados, limite do Yahoo...
            print(f"-> erro: {erro}")
            falharam.append(ticker)
            continue
        if tabela.empty:
            print("-> sem dados")
            falharam.append(ticker)
            continue
        print(f"-> {len(tabela)} pregões")
        tabelas.append(tabela)
        if pausa:
            time.sleep(pausa)  # pausa curta para o Yahoo não bloquear

    if not tabelas:
        return pd.DataFrame(columns=COLUNAS), falharam
    precos = pd.concat(tabelas, ignore_index=True)
    precos = remover_datas_repetidas(remover_pregoes_incompletos(precos))
    return precos[COLUNAS].sort_values(["ticker", "data"]).reset_index(drop=True), falharam
