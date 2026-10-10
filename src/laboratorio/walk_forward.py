"""
walk_forward.py
Treina de novo no primeiro pregão de cada mês, usando só o passado, com EMBARGO.

Por que o embargo: o rótulo do dia D só fica conhecido no fechamento de D+10. Um treino feito
no dia T só pode usar linhas com D ≤ calendario[pos(T) − 10]. Sem isso, o modelo aprenderia
com resultados que ainda não tinham acontecido.
"""

import pandas as pd

import config
from src.features.preco import COLUNAS_MODELO
from src.modelos import combinador


def datas_de_retreino(calendario: pd.DatetimeIndex, inicio, fim) -> list[pd.Timestamp]:
    dias = calendario[(calendario >= pd.Timestamp(inicio)) & (calendario <= pd.Timestamp(fim))]
    return list(pd.Series(dias).groupby(dias.to_period("M")).first())


def limite_de_treino(calendario: pd.DatetimeIndex, dia, horizonte: int = config.HORIZONTE):
    """Último dia cujo rótulo já era conhecido no fechamento de 'dia'. None se não houver."""
    posicao = calendario.get_loc(pd.Timestamp(dia)) - horizonte
    return calendario[posicao] if posicao >= 0 else None


def ultimo_dia_avaliavel(calendario: pd.DatetimeIndex, fim, horizonte: int = config.HORIZONTE):
    """Último dia D cujo resultado (D + horizonte) cai até 'fim'. None se não houver."""
    dias = calendario[calendario <= pd.Timestamp(fim)]
    posicao = len(dias) - 1 - horizonte
    return dias[posicao] if posicao >= 0 else None


def gerar_probabilidades(tabela: pd.DataFrame, calendario: pd.DatetimeIndex, inicio, fim, tipo: str,
                         colunas: list[str] = COLUNAS_MODELO, horizonte: int = config.HORIZONTE,
                         treinar=combinador.treinar, prob=combinador.prob_superar) -> pd.DataFrame:
    """Probabilidade de cada (dia, ação) entre 'inicio' e 'fim', com retreino mensal."""
    inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    retreinos = datas_de_retreino(calendario, inicio, fim)
    partes = []
    for i, dia in enumerate(retreinos):
        proximo = retreinos[i + 1] if i + 1 < len(retreinos) else fim + pd.Timedelta(days=1)
        limite = limite_de_treino(calendario, dia, horizonte)
        if limite is None:
            continue
        treino = tabela[(tabela["data"] <= limite) & tabela["rotulo"].notna()]
        alvo = tabela[(tabela["data"] >= dia) & (tabela["data"] < proximo) & (tabela["data"] <= fim)]
        if treino.empty or alvo.empty or treino["rotulo"].nunique() < 2:
            continue
        modelo = treinar(treino[colunas], treino["rotulo"].astype(int), tipo)
        partes.append(pd.DataFrame({"data": alvo["data"].values, "ticker": alvo["ticker"].values,
                                    "prob": prob(modelo, alvo[colunas])}))
    if not partes:
        return pd.DataFrame(columns=["data", "ticker", "prob"])
    return pd.concat(partes, ignore_index=True)
