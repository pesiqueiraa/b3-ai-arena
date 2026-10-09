"""
base.py
Junta as peças da etapa 1 numa única "base" usada pelo laboratório, pelos sinais e pela prova final.

    tabela:     uma linha por (dia, ação do universo) com as features e o alvo
                (rotulo = NaN quando o futuro ainda não aconteceu)
    calendario: os pregões
    retornos:   retorno diário de cada ação (para simular a carteira)
"""

from dataclasses import dataclass

import pandas as pd

import config
from src.alvo import calcular_alvo, retornos_diarios
from src.features.preco import adicionar_ranks, calcular_features
from src.universo import membros_por_dia


@dataclass
class Base:
    tabela: pd.DataFrame
    calendario: pd.DatetimeIndex
    retornos: pd.DataFrame


def montar_calendario(datas_b3: pd.Series, datas_referencia: pd.Series) -> pd.DatetimeIndex:
    """
    Pregões = datas da B3 + datas do BOVA11 no Yahoo. A união cobre o caso de o arquivo anual
    da B3 do ano corrente ainda não ter os últimos dias.
    """
    datas = pd.concat([pd.Series(datas_b3), pd.Series(datas_referencia)]).drop_duplicates().sort_values()
    datas = datas[datas >= pd.Timestamp(config.DATA_DOWNLOAD_INICIO)]
    return pd.DatetimeIndex(datas.values)


def montar_base(precos: pd.DataFrame, universo: pd.DataFrame, calendario: pd.DatetimeIndex) -> Base:
    feat = calcular_features(precos, calendario)
    feat = adicionar_ranks(feat, membros_por_dia(universo, calendario))
    alvo = calcular_alvo(precos, calendario)
    tabela = feat.merge(alvo, on=["data", "ticker"], how="left")
    return Base(tabela=tabela, calendario=calendario, retornos=retornos_diarios(precos, calendario))


def carregar_base() -> Base:
    """Lê os arquivos gerados por 'python main.py atualizar' e monta a base."""
    for arquivo in (config.ARQUIVO_PRECOS, config.ARQUIVO_UNIVERSO, config.ARQUIVO_CALENDARIO):
        if not arquivo.exists():
            raise SystemExit(f"Não achei {arquivo.name}. Rode antes: python main.py atualizar")
    precos = pd.read_csv(config.ARQUIVO_PRECOS, parse_dates=["data"])
    universo = pd.read_csv(config.ARQUIVO_UNIVERSO, parse_dates=["data_inicio"], dtype={"mes": str})
    calendario = pd.DatetimeIndex(pd.read_csv(config.ARQUIVO_CALENDARIO, parse_dates=["data"])["data"])
    return montar_base(precos, universo, calendario)
