"""
features.py
Quais informações o ML pode ver, e uma função que entrega SÓ essas informações.

O gabarito nunca entra aqui. Se alguém tentar incluir, o programa para com erro.
"""

import numpy as np
import pandas as pd

# Indicadores que o modelo pode usar. Todos olham só para o passado (ver src/dados.py).
FEATURES = [
    "retorno_1d", "retorno_5d", "retorno_20d",
    "distancia_media_20d", "volatilidade_20d", "volume_relativo_20d",
]

# Qualquer coluna com "gabarito" no nome é a resposta da prova.
assert not any("gabarito" in f for f in FEATURES), "O gabarito não pode ser feature!"


def extrair_features(linhas: pd.DataFrame) -> pd.DataFrame:
    """
    Devolve uma tabela só com as colunas de FEATURES (sem data, sem ticker, sem gabarito).
    Divisões por zero viram "vazio" (NaN), que o modelo sabe preencher.
    """
    x = linhas[FEATURES].copy()
    return x.replace([np.inf, -np.inf], np.nan)
