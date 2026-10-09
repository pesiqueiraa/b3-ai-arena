"""
preco.py
Features de swing trade calculadas SÓ com preços e volumes até o fechamento de D.
Toda coluna começa com "p_" (de preço); na etapa 3 virão as "t_" (de textos).

    p_ret_N        retorno em N pregões (5, 20, 60, 120)
    p_dist_mmN     distância do preço até a média de N pregões (20, 50)
    p_vol_N        volatilidade diária em N pregões (20, 60)
    p_volume_rel_20  volume de hoje / média de 20 pregões
    p_rel_bova_N   retorno em N pregões menos o do BOVA11 (20, 60)

A ação precisa de 120 pregões de histórico (a p_ret_120 precisa deles); antes disso, fica de fora.
Os "_rank" dão a posição da ação entre as do universo NAQUELE DIA (0 a 1), o que deixa
comparáveis anos calmos e anos agitados.
"""

import numpy as np
import pandas as pd

import config

COLUNAS = [
    "p_ret_5", "p_ret_20", "p_ret_60", "p_ret_120",
    "p_dist_mm20", "p_dist_mm50",
    "p_vol_20", "p_vol_60",
    "p_volume_rel_20",
    "p_rel_bova_20", "p_rel_bova_60",
]
COLUNAS_MODELO = COLUNAS + [f"{c}_rank" for c in COLUNAS]


def _de_uma_acao(g: pd.DataFrame) -> pd.DataFrame:
    """g: uma ação ordenada por data, com fechamento_ajustado, volume e bova (preço do BOVA11 no dia)."""
    preco = g["fechamento_ajustado"]
    bova = g["bova"]
    volume = g["volume"].replace(0, np.nan)
    retorno = preco / preco.shift(1) - 1

    saida = g[["data", "ticker"]].copy()
    for n in (5, 20, 60, 120):
        saida[f"p_ret_{n}"] = preco / preco.shift(n) - 1
    saida["p_dist_mm20"] = preco / preco.rolling(20).mean() - 1
    saida["p_dist_mm50"] = preco / preco.rolling(50).mean() - 1
    saida["p_vol_20"] = retorno.rolling(20).std()
    saida["p_vol_60"] = retorno.rolling(60).std()
    saida["p_volume_rel_20"] = volume / volume.rolling(20, min_periods=10).mean()
    for n in (20, 60):
        saida[f"p_rel_bova_{n}"] = saida[f"p_ret_{n}"] - (bova / bova.shift(n) - 1)
    return saida


def calcular_features(precos: pd.DataFrame, calendario: pd.DatetimeIndex) -> pd.DataFrame:
    precos = precos[precos["data"].isin(calendario)]
    bova = (precos.loc[precos["ticker"] == config.REFERENCIA, ["data", "fechamento_ajustado"]]
                  .rename(columns={"fechamento_ajustado": "bova"}))
    acoes = precos[precos["ticker"] != config.REFERENCIA]
    partes = [_de_uma_acao(g.sort_values("data").merge(bova, on="data", how="left"))
              for _, g in acoes.groupby("ticker")]
    if not partes:
        return pd.DataFrame(columns=["data", "ticker", *COLUNAS])
    feat = pd.concat(partes, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(subset=["p_ret_120"])  # 120 pregões de histórico
    return feat.sort_values(["data", "ticker"]).reset_index(drop=True)


def adicionar_ranks(feat: pd.DataFrame, membros: pd.DataFrame) -> pd.DataFrame:
    """Mantém só as ações do universo de cada dia e acrescenta a posição de cada feature (0 a 1)."""
    base = feat.merge(membros, on=["data", "ticker"], how="inner")
    for coluna in COLUNAS:
        base[f"{coluna}_rank"] = base.groupby("data")[coluna].rank(pct=True)
    return base.sort_values(["data", "ticker"]).reset_index(drop=True)
