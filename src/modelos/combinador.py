"""
combinador.py
O ML que junta todos os sinais e aprende no histórico quanto cada um vale.

    "logistica": regressão logística (a régua simples e transparente)
    "boosting":  gradient boosting de histogramas (mais flexível; aceita valores vazios)

As funções recebem SÓ features (e, no treino, o rótulo dos dias passados). Datas e embargo
ficam em src/laboratorio/walk_forward.py.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


def criar_modelo(tipo: str):
    if tipo == "logistica":
        # Imputação pela mediana do TREINO, normalização e logística: tudo aprende só com o treino.
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    if tipo == "boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=200, random_state=config.SEMENTE)
    raise ValueError(f"tipo de modelo desconhecido: '{tipo}' (use um de {config.TIPOS_MODELO})")


def treinar(x: pd.DataFrame, y: pd.Series, tipo: str):
    modelo = criar_modelo(tipo)
    modelo.fit(x, y)
    return modelo


def prob_superar(modelo, x: pd.DataFrame) -> np.ndarray:
    """Probabilidade (0 a 1) de cada ação superar o BOVA11 nos próximos 10 pregões."""
    return modelo.predict_proba(x)[:, 1]
