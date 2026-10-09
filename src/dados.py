"""
dados.py  (vem do antigo baixar_acoes.py)
Baixa o histórico diário de ações da B3 (via Yahoo Finance) e calcula os indicadores.

Gera dois arquivos na pasta "dados/":
    precos_b3.csv       -> preços brutos, uma linha por ação por dia
    indicadores_b3.csv  -> indicadores calculados + o gabarito (retorno do dia seguinte)

Dica: rode depois que o pregão fecha (após ~18h) para o dia de hoje vir completo.
"""

import time

import pandas as pd

import config

COLUNAS_PRECO = ["abertura", "maxima", "minima", "fechamento", "fechamento_ajustado", "volume"]


# ---------------------------------------------------------------------------
# 1) DOWNLOAD
# ---------------------------------------------------------------------------

def baixar_ticker(ticker: str, inicio: str) -> pd.DataFrame:
    """Baixa os preços diários de uma ação e devolve uma tabela com colunas em português."""
    import yfinance as yf  # importado aqui para os testes rodarem sem internet

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


def remover_pregoes_incompletos(precos: pd.DataFrame) -> pd.DataFrame:
    """
    CORREÇÃO (não existia no script original).
    Quando o download acontece antes de o Yahoo consolidar o dia, a linha de hoje vem
    com abertura, máxima e mínima iguais a zero e volume zero. Esse "pregão pela metade"
    deixaria os indicadores do dia errados (ou vazios). Removemos essas linhas;
    basta baixar de novo mais tarde para o dia entrar completo.

    Atenção: linhas antigas com volume 0 mas preços preenchidos (ex.: BBAS3 e VALE3
    em 26/04/2024) NÃO são removidas. São dias sem negócio registrado, não dias incompletos.
    """
    incompleta = (precos["abertura"] == 0) & (precos["maxima"] == 0) & (precos["minima"] == 0)
    if incompleta.any():
        datas = sorted(precos.loc[incompleta, "data"].dt.strftime("%d/%m/%Y").unique())
        print(f"Aviso: {int(incompleta.sum())} linha(s) de pregão incompleto removidas "
              f"(datas: {', '.join(datas)}). Baixe de novo depois do fechamento.")
    return precos[~incompleta].copy()


# ---------------------------------------------------------------------------
# 2) INDICADORES (a base do "estado" que os modelos vão receber)
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
    # Serve só para dar nota às decisões depois (e, no ML, como resposta certa dos
    # dias PASSADOS no treino). NUNCA mande esta coluna para o Jev, para o LLM ou
    # como feature do ML: seria entregar a resposta da prova.
    # Na última linha de cada ação ela fica vazia, porque "amanhã" ainda não aconteceu.
    df["gabarito_retorno_amanha"] = preco.shift(-1) / preco - 1
    return df


def calcular_indicadores(precos: pd.DataFrame, data_corte: str = config.DATA_CORTE_JEV) -> pd.DataFrame:
    partes = [indicadores_de_uma_acao(df) for _, df in precos.groupby("ticker")]
    ind = pd.concat(partes, ignore_index=True)

    # Marca em qual período cada linha está.
    ind["periodo"] = "historico"
    ind.loc[ind["data"] >= pd.Timestamp(data_corte), "periodo"] = "teste_limpo"

    # Os primeiros ~20 pregões de cada ação não têm passado suficiente para os indicadores.
    ind = ind.dropna(subset=["retorno_20d", "volatilidade_20d"])
    return ind.sort_values(["data", "ticker"]).reset_index(drop=True)


def carregar_indicadores() -> pd.DataFrame:
    """Lê dados/indicadores_b3.csv (criado pela etapa 'baixar')."""
    if not config.ARQUIVO_INDICADORES.exists():
        raise SystemExit(f"Não achei {config.ARQUIVO_INDICADORES}. Rode antes: python main.py baixar")
    return pd.read_csv(config.ARQUIVO_INDICADORES, parse_dates=["data"])


# ---------------------------------------------------------------------------
# 3) ETAPA "baixar"
# ---------------------------------------------------------------------------

def rodar() -> None:
    config.PASTA_DADOS.mkdir(exist_ok=True)

    tabelas, falharam = [], []
    for ticker in config.TICKERS:
        print(f"Baixando {ticker:<7}", end=" ", flush=True)
        try:
            tabela = baixar_ticker(ticker, config.DATA_INICIO)
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
    precos = remover_pregoes_incompletos(precos)
    indicadores = calcular_indicadores(precos)

    precos.to_csv(config.ARQUIVO_PRECOS, index=False, date_format="%Y-%m-%d")
    indicadores.to_csv(config.ARQUIVO_INDICADORES, index=False, date_format="%Y-%m-%d", float_format="%.6f")

    limpo = indicadores[indicadores["periodo"] == "teste_limpo"]
    print("\n===== Resumo =====")
    print(f"Ações baixadas: {precos['ticker'].nunique()} de {len(config.TICKERS)}")
    print(f"De {precos['data'].min():%d/%m/%Y} até {precos['data'].max():%d/%m/%Y}")
    print(f"Pregões na janela limpa (desde {config.DATA_CORTE_JEV}): {limpo['data'].nunique()}")
    print(f"Arquivos salvos: {config.ARQUIVO_PRECOS.name} e {config.ARQUIVO_INDICADORES.name} em {config.PASTA_DADOS}")
    if falharam:
        print(f"Atenção, não vieram: {', '.join(falharam)}")

    print("\nExemplo (último dia de PETR4):")
    exemplo = indicadores[indicadores["ticker"] == "PETR4"].tail(1).copy()
    if not exemplo.empty:
        exemplo["data"] = exemplo["data"].dt.strftime("%d/%m/%Y")
        print(exemplo.T.to_string(header=False))
