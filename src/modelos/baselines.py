"""
baselines.py
Modelos "bobos" que servem de régua. Qualquer modelo esperto precisa bater estes.

(A outra régua, BOVA11 comprar e segurar, é calculada direto em avaliacao.py.
 Os agentes aleatórios também ficam lá, no teste contra a sorte.)
"""

import pandas as pd

import config
from src import dados, decisoes


def sempre_comprar(linhas: pd.DataFrame) -> pd.DataFrame:
    """Compra todas as ações todo dia (carteira igualitária). Recebe só data e ticker."""
    saida = linhas[["data", "ticker"]].copy()
    saida["prob_subir"] = 1.0
    saida["decisao"] = "comprar"
    return saida


def rodar(refazer: bool = False) -> None:
    df = dados.carregar_indicadores()
    df = df[(df["ticker"] != config.REFERENCIA) & (df["periodo"] == "teste_limpo")]

    nome = "sempre_comprar"
    dias = sorted(df["data"].unique())
    pendentes = decisoes.dias_pendentes(nome, dias, refazer)
    if not pendentes:
        print("sempre_comprar: nada novo a decidir.")
        return
    decisoes.acrescentar(sempre_comprar(df[df["data"].isin(pendentes)]), nome, refazer)
