"""
alvo.py
O que a linha tenta prever e quanto a carteira ganhou de fato.

    excesso(i, D) = retorno ajustado de i entre D e D+10 − retorno ajustado do BOVA11 entre D e D+10
    rotulo        = 1 se excesso > 0, senão 0

O rótulo de D só fica conhecido no fechamento de D+10. Por isso o treino usa EMBARGO
(ver src/laboratorio/walk_forward.py). Ele nunca entra nas features.
"""

import pandas as pd

import config


def _larga(precos: pd.DataFrame, calendario: pd.DatetimeIndex) -> pd.DataFrame:
    """Tabela larga: uma linha por pregão do calendário, uma coluna por ticker."""
    return (precos.pivot_table(index="data", columns="ticker", values="fechamento_ajustado", aggfunc="last")
                  .reindex(calendario))


def calcular_alvo(precos: pd.DataFrame, calendario: pd.DatetimeIndex,
                  horizonte: int = config.HORIZONTE) -> pd.DataFrame:
    larga = _larga(precos, calendario)
    if config.REFERENCIA not in larga.columns:
        raise ValueError(f"faltam os preços do {config.REFERENCIA} (a referência do alvo).")
    futuro = larga.shift(-horizonte) / larga - 1          # NaN se faltar preço em D ou em D+h
    excesso = futuro.drop(columns=config.REFERENCIA).sub(futuro[config.REFERENCIA], axis=0)
    longo = (excesso.rename_axis(index="data", columns="ticker").reset_index()
                    .melt(id_vars="data", var_name="ticker", value_name="excesso")
                    .dropna(subset=["excesso"]))
    longo["rotulo"] = (longo["excesso"] > 0).astype(int)
    return longo.sort_values(["data", "ticker"]).reset_index(drop=True)


def retornos_diarios(precos: pd.DataFrame, calendario: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Retorno de cada pregão para a simulação da carteira. Repete o último preço conhecido; depois
    que a ação some, o retorno fica 0 (a posição é encerrada no último preço disponível).
    """
    larga = _larga(precos, calendario).ffill()
    return (larga / larga.shift(1) - 1).fillna(0.0)
