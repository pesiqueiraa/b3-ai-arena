"""
sinais.py
Transforma probabilidades em sinais.

    compra: as k ações de MAIOR probabilidade de superar o BOVA11, se prob ≥ p_min
    venda:  as k ações de MENOR probabilidade, se prob ≤ 1 − p_min
            ("deve ficar para trás"; não é recomendação de vender a descoberto)

Uma ação nunca é compra e venda no mesmo dia (pode acontecer num empate exato com p_min = 0,5).
"""

import pandas as pd

COLUNAS = ["data", "ticker", "prob", "sinal"]


def gerar_sinais(probs: pd.DataFrame, k: int, p_min: float) -> pd.DataFrame:
    if probs.empty:
        return pd.DataFrame(columns=COLUNAS)
    topo = probs.sort_values(["data", "prob"], ascending=[True, False]).groupby("data").head(k)
    compras = topo[topo["prob"] >= p_min].assign(sinal="compra")
    fundo = probs.sort_values(["data", "prob"], ascending=[True, True]).groupby("data").head(k)
    vendas = fundo[fundo["prob"] <= 1 - p_min].assign(sinal="venda")
    vendas = vendas.merge(compras[["data", "ticker"]], on=["data", "ticker"], how="left", indicator=True)
    vendas = vendas[vendas["_merge"] == "left_only"].drop(columns="_merge")
    sinais = pd.concat([compras, vendas], ignore_index=True)[COLUNAS]
    return sinais.reset_index(drop=True)
