"""
walk_forward.py
O treino "walk-forward": a cada dia D, o modelo é treinado de novo usando SÓ o passado.

Este é o ÚNICO arquivo do ML que encosta na coluna de gabarito, e só de duas formas:
  1. para filtrar as linhas de treino (dias ANTERIORES a D com gabarito conhecido);
  2. para virar a resposta certa do treino: subiu = gabarito > 0.
A linha do próprio dia D chega ao modelo só com as features (ver features.py).

Por que a linha de D-1 pode entrar no treino? O gabarito dela é o retorno de D-1 para D,
que já é conhecido no fechamento de D, que é quando a decisão é tomada.
"""

import numpy as np
import pandas as pd

import config
from src.modelos.ml import modelo as ml_modelo
from src.modelos.ml.features import FEATURES, extrair_features

COLUNA_GABARITO = "gabarito_retorno_amanha"


def decidir_um_dia(df: pd.DataFrame, dia, treinar=ml_modelo.treinar, prob_subir=ml_modelo.prob_subir) -> pd.DataFrame:
    """
    Gera as decisões de um único dia D.
    'treinar' e 'prob_subir' podem ser trocados nos testes, para espiar o que o modelo recebe.
    """
    treino = df[df["data"] < dia].dropna(subset=FEATURES + [COLUNA_GABARITO])
    hoje = df[df["data"] == dia]
    if treino.empty or hoje.empty:
        return pd.DataFrame(columns=["data", "ticker", "prob_subir", "decisao"])

    subiu = (treino[COLUNA_GABARITO] > 0).astype(int)
    modelo = treinar(extrair_features(treino), subiu)
    prob = prob_subir(modelo, extrair_features(hoje))

    return pd.DataFrame({
        "data": hoje["data"].values,
        "ticker": hoje["ticker"].values,
        "prob_subir": prob,
        "decisao": np.where(prob > config.LIMIAR_COMPRA, "comprar", "ficar_de_fora"),
    })


def gerar_decisoes(df: pd.DataFrame, dias: list, **kwargs) -> pd.DataFrame:
    """Roda o walk-forward em cada dia da lista e junta tudo numa tabela."""
    partes = [decidir_um_dia(df, dia, **kwargs) for dia in dias]
    partes = [p for p in partes if not p.empty]
    if not partes:
        return pd.DataFrame(columns=["data", "ticker", "prob_subir", "decisao"])
    return pd.concat(partes, ignore_index=True)
