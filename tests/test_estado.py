"""
Testes do estado em texto (src/estado.py), o que o Jev e o LLM recebem.

Rodar:  python -m pytest
"""

import re

import numpy as np
import pandas as pd
import pytest

from src import estado
from src.modelos.ml.features import FEATURES


def _linha(**mudancas) -> pd.Series:
    base = {
        "data": pd.Timestamp("2026-10-08"), "ticker": "PETR4",
        "retorno_1d": 0.021167, "retorno_5d": 0.111824, "retorno_20d": 0.129479,
        "distancia_media_20d": 0.10475, "volatilidade_20d": 0.025144,
        "volume_relativo_20d": 1.130526, "gabarito_retorno_amanha": 0.0345, "periodo": "teste_limpo",
    }
    base.update(mudancas)
    return pd.Series(base)


def test_texto_completo_do_exemplo():
    assert estado.gerar_estado(_linha()) == (
        "PETR4 (Petrobras), pregão de 08/10/2026. Retorno no último pregão: +2,1%. "
        "Retorno em 5 pregões: +11,2%. Retorno em 20 pregões: +12,9%. "
        "Preço 10,5% acima da média de 20 pregões. "
        "Oscilação diária típica nos últimos 20 pregões: 2,5%. Volume 13% acima do normal."
    )


def test_estado_usa_as_mesmas_informacoes_do_ml():
    """Regra 4 + comparação justa: Jev/LLM e ML veem os mesmos seis indicadores."""
    assert estado.ESTADO_COLUNAS == FEATURES


def test_gabarito_nunca_entra_no_texto():
    """Mudar ou apagar o gabarito não pode mudar uma letra do estado."""
    a = estado.gerar_estado(_linha(gabarito_retorno_amanha=0.0345))
    b = estado.gerar_estado(_linha(gabarito_retorno_amanha=-0.20))
    c = estado.gerar_estado(_linha().drop(["gabarito_retorno_amanha", "periodo"]))
    assert a == b == c
    assert "3,5" not in a and "20,0" not in b


def test_gerar_estados_descarta_o_gabarito_logo_na_entrada():
    df = pd.DataFrame([_linha(), _linha(ticker="VALE3")])
    saida = estado.gerar_estados(df)
    assert list(saida.columns) == ["data", "ticker", "estado"]
    sem_gabarito = estado.gerar_estados(df.drop(columns=["gabarito_retorno_amanha"]))
    pd.testing.assert_frame_equal(saida, sem_gabarito)


def test_anonimizado_nao_tem_ticker_nome_nem_data():
    texto = estado.gerar_estado(_linha(), anonimizar=True)
    assert texto.startswith("Ação A.")
    for proibido in ["PETR4", "Petrobras", "2026", "08/10", "pregão de"]:
        assert proibido not in texto
    assert not re.search(r"\d{2}/\d{2}", texto)


def test_anonimizado_tem_os_mesmos_numeros():
    """Só o cabeçalho muda: o teste de memorização compara exatamente a mesma informação."""
    normal = estado.gerar_estado(_linha())
    anonimo = estado.gerar_estado(_linha(), anonimizar=True)
    assert normal.split(". ", 1)[1] == anonimo.split(". ", 1)[1]


@pytest.mark.parametrize("fracao, esperado", [
    (0.038, "+3,8%"), (-0.038, "-3,8%"), (0.0, "0,0%"), (-0.00001, "0,0%"), (0.00049, "0,0%"),
])
def test_formatar_pct(fracao, esperado):
    assert estado.formatar_pct(fracao) == esperado


@pytest.mark.parametrize("campo, valor, trecho", [
    ("distancia_media_20d", -0.068, "Preço 6,8% abaixo da média de 20 pregões."),
    ("distancia_media_20d", 0.0001, "Preço igual à média de 20 pregões."),
    ("volume_relativo_20d", 0.70, "Volume 30% abaixo do normal."),
    ("volume_relativo_20d", 1.30, "Volume 30% acima do normal."),
    ("volume_relativo_20d", 1.002, "Volume igual ao normal."),
    ("volume_relativo_20d", 2.3018, "Volume 2,3 vezes o normal."),
    ("volume_relativo_20d", np.nan, "Volume: sem dado."),
    ("retorno_5d", np.nan, "Retorno em 5 pregões: sem dado."),
])
def test_casos_limite(campo, valor, trecho):
    assert trecho in estado.gerar_estado(_linha(**{campo: valor}))


def test_mesmo_dado_gera_sempre_o_mesmo_texto(acoes):
    """Determinismo: o cache das APIs depende de o texto ser idêntico entre execuções."""
    amostra = acoes.sample(50, random_state=0)
    assert estado.gerar_estados(amostra).equals(estado.gerar_estados(amostra))
    assert estado.gerar_estados(amostra)["estado"].is_unique
