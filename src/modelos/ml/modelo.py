"""
modelo.py
O modelo de ML em si: uma regressão logística simples.

Estas funções recebem SÓ features (e, no treino, a resposta certa dos dias passados).
Elas não sabem nada de datas nem de gabarito: quem cuida disso é o walk_forward.py.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def criar_modelo():
    """
    Regressão logística: um modelo simples e honesto como primeira referência.

    Tudo fica dentro de um "pipeline", então cada etapa aprende SÓ com os dados de treino:
    - SimpleImputer: se um indicador vier vazio no dia da decisão (ex.: volume zerado),
      preenche com a mediana do TREINO. CORREÇÃO: antes, a linha era descartada em
      silêncio e o modelo simplesmente não decidia naquele dia/ação.
    - StandardScaler: coloca os indicadores na mesma escala.
    - LogisticRegression: estima a probabilidade de a ação subir.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def treinar(x_treino: pd.DataFrame, subiu: pd.Series):
    """Treina um modelo novo. 'subiu' vale 1 quando a ação subiu no pregão seguinte."""
    modelo = criar_modelo()
    modelo.fit(x_treino, subiu)
    return modelo


def prob_subir(modelo, x_hoje: pd.DataFrame):
    """Probabilidade (0 a 1) de cada ação subir no próximo pregão."""
    return modelo.predict_proba(x_hoje)[:, 1]
