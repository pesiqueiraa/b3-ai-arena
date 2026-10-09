"""
metricas.py
As métricas do relatório do laboratório e da prova final.

    brier             erro médio da probabilidade (menor é melhor; dizer sempre 50% dá 0,25)
    prob_referencia   probabilidade "boba": fração de rótulos 1 até D − 10 (só o que era conhecido)
    acerto_sinais     % das compras que superaram o BOVA11 (ou das vendas que ficaram para trás)
    resumo_carteira   retorno, excesso sobre o BOVA11, anos bons e pior queda
    bateu_aleatorios  % de agentes que, comprando o MESMO número de ações por dia sorteadas
                      do universo do dia, a linha superou (pelo excesso médio em 10 pregões)
"""

import numpy as np
import pandas as pd

import config


def brier(prob, rotulo) -> float:
    return float(np.mean((np.asarray(prob, dtype=float) - np.asarray(rotulo, dtype=float)) ** 2))


def prob_referencia(tabela: pd.DataFrame, calendario: pd.DatetimeIndex,
                    horizonte: int = config.HORIZONTE) -> pd.Series:
    conhecidos = tabela.dropna(subset=["rotulo"])
    por_dia = (conhecidos.groupby("data")["rotulo"].agg(["sum", "count"])
                         .reindex(calendario, fill_value=0).cumsum())
    taxa = por_dia["sum"] / por_dia["count"].replace(0, np.nan)
    return taxa.shift(horizonte)  # no dia D, só vale o que era conhecido até D − horizonte


def acerto_sinais(sinais: pd.DataFrame, tabela: pd.DataFrame, sinal: str) -> float:
    m = (sinais[sinais["sinal"] == sinal]
         .merge(tabela[["data", "ticker", "excesso"]], on=["data", "ticker"])
         .dropna(subset=["excesso"]))
    if m.empty:
        return float("nan")
    certos = m["excesso"] > 0 if sinal == "compra" else m["excesso"] < 0
    return float(certos.mean())


def acumulado(ret: pd.Series) -> float:
    return float((1 + ret).prod() - 1)


def pior_queda(ret: pd.Series) -> float:
    curva = (1 + ret).cumprod()
    return float((curva / curva.cummax() - 1).min()) if len(curva) else float("nan")


def resumo_carteira(ret: pd.Series, ret_bova: pd.Series) -> dict:
    anos = ret.groupby(ret.index.year).apply(acumulado)
    anos_bova = ret_bova.groupby(ret_bova.index.year).apply(acumulado)
    excesso_anos = anos - anos_bova
    return {
        "retorno_total": acumulado(ret),
        "retorno_bova11": acumulado(ret_bova),
        "excesso_total": acumulado(ret) - acumulado(ret_bova),
        "excesso_medio_anual": float(excesso_anos.mean()) if len(excesso_anos) else float("nan"),
        "fracao_anos_bons": float((excesso_anos > 0).mean()) if len(excesso_anos) else float("nan"),
        "pior_queda": pior_queda(ret),
    }


def bateu_aleatorios(sinais: pd.DataFrame, tabela: pd.DataFrame, n_agentes: int, rng) -> float:
    conhecidos = tabela.dropna(subset=["excesso"])
    compras = (sinais[sinais["sinal"] == "compra"]
               .merge(conhecidos[["data", "ticker", "excesso"]], on=["data", "ticker"]))
    if compras.empty:
        return float("nan")
    excessos_do_dia = conhecidos.groupby("data")["excesso"].apply(np.asarray)
    nota_da_linha = compras.groupby("data")["excesso"].mean().mean()

    soma = np.zeros(n_agentes)
    dias = 0
    for dia, k in compras.groupby("data").size().items():
        opcoes = excessos_do_dia[dia]
        k = min(k, len(opcoes))
        sorteio = np.argsort(rng.random((n_agentes, len(opcoes))), axis=1)[:, :k]
        soma += opcoes[sorteio].mean(axis=1)
        dias += 1
    return float(np.mean(nota_da_linha > soma / dias))
