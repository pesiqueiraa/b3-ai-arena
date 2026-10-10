"""
Testes do estado em texto (src/estado.py), o que o Jev e o LLM recebem.
O experimento usa inglês (config.ESTADO_IDIOMA); o português continua disponível e testado.

Rodar:  python -m pytest
"""

import re

import numpy as np
import pandas as pd
import pytest

import config
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


def test_experimento_usa_ingles():
    """Desvio 1 do PREREGISTRO.md: inglês é o idioma principal do Jev."""
    assert config.ESTADO_IDIOMA == "en"
    assert estado.PERGUNTA == "Will the stock close the next trading session above today's close?"


def test_texto_completo_em_ingles():
    assert estado.gerar_estado(_linha()) == (
        "PETR4 (Petrobras), trading session of October 8, 2026. Return in the last session: +2.1%. "
        "Return over the last 5 sessions: +11.2%. Return over the last 20 sessions: +12.9%. "
        "Price 10.5% above its 20-session average. "
        "Typical daily move over the last 20 sessions: 2.5%. Volume 13% above normal."
    )


def test_texto_completo_em_portugues_igual_ao_da_v1():
    """O português precisa continuar idêntico ao que o LLM recebeu no prompt v1."""
    assert estado.gerar_estado(_linha(), idioma="pt") == (
        "PETR4 (Petrobras), pregão de 08/10/2026. Retorno no último pregão: +2,1%. "
        "Retorno em 5 pregões: +11,2%. Retorno em 20 pregões: +12,9%. "
        "Preço 10,5% acima da média de 20 pregões. "
        "Oscilação diária típica nos últimos 20 pregões: 2,5%. Volume 13% acima do normal."
    )


def test_os_dois_idiomas_tem_os_mesmos_numeros():
    """Tradução direta: os números (sem a data) são os mesmos, só muda a vírgula decimal."""
    numeros = lambda t: re.findall(r"[+-]?\d+[.,]?\d*%|\d+[.,]\d+ (?:times|vezes)", t)
    en = [n.replace(",", ".").replace("vezes", "times") for n in numeros(estado.gerar_estado(_linha(), idioma="en"))]
    pt = [n.replace(",", ".").replace("vezes", "times") for n in numeros(estado.gerar_estado(_linha(), idioma="pt"))]
    assert en == pt and len(en) == 6


def test_estado_usa_as_mesmas_informacoes_do_ml():
    """Regra 4 + comparação justa: Jev/LLM e ML veem os mesmos seis indicadores."""
    assert estado.ESTADO_COLUNAS == FEATURES


@pytest.mark.parametrize("idioma", ["en", "pt"])
def test_gabarito_nunca_entra_no_texto(idioma):
    """Mudar ou apagar o gabarito não pode mudar uma letra do estado."""
    a = estado.gerar_estado(_linha(gabarito_retorno_amanha=0.0345), idioma=idioma)
    b = estado.gerar_estado(_linha(gabarito_retorno_amanha=-0.20), idioma=idioma)
    c = estado.gerar_estado(_linha().drop(["gabarito_retorno_amanha", "periodo"]), idioma=idioma)
    assert a == b == c


def test_gerar_estados_descarta_o_gabarito_logo_na_entrada():
    df = pd.DataFrame([_linha(), _linha(ticker="VALE3")])
    saida = estado.gerar_estados(df)
    assert list(saida.columns) == ["data", "ticker", "estado"]
    sem_gabarito = estado.gerar_estados(df.drop(columns=["gabarito_retorno_amanha"]))
    pd.testing.assert_frame_equal(saida, sem_gabarito)


@pytest.mark.parametrize("idioma, rotulo", [("en", "Stock A."), ("pt", "Ação A.")])
def test_anonimizado_nao_tem_ticker_nome_nem_data(idioma, rotulo):
    texto = estado.gerar_estado(_linha(), anonimizar=True, idioma=idioma)
    assert texto.startswith(rotulo)
    for proibido in ["PETR4", "Petrobras", "2026", "October", "08/10", "session of", "pregão de"]:
        assert proibido not in texto
    assert not re.search(r"\d{2}/\d{2}", texto)


def test_anonimizado_tem_os_mesmos_numeros():
    """Só o cabeçalho muda: o teste de memorização compara exatamente a mesma informação."""
    normal = estado.gerar_estado(_linha())
    anonimo = estado.gerar_estado(_linha(), anonimizar=True)
    assert normal.split(". ", 1)[1] == anonimo.split(". ", 1)[1]


@pytest.mark.parametrize("fracao, idioma, esperado", [
    (0.038, "pt", "+3,8%"), (-0.038, "pt", "-3,8%"), (0.0, "pt", "0,0%"), (-0.00001, "pt", "0,0%"),
    (0.00049, "pt", "0,0%"), (0.038, "en", "+3.8%"), (-0.038, "en", "-3.8%"), (-0.00001, "en", "0.0%"),
])
def test_formatar_pct(fracao, idioma, esperado):
    assert estado.formatar_pct(fracao, idioma=idioma) == esperado


@pytest.mark.parametrize("campo, valor, idioma, trecho", [
    ("distancia_media_20d", -0.068, "en", "Price 6.8% below its 20-session average."),
    ("distancia_media_20d", 0.0001, "en", "Price equal to its 20-session average."),
    ("volume_relativo_20d", 0.70, "en", "Volume 30% below normal."),
    ("volume_relativo_20d", 1.30, "en", "Volume 30% above normal."),
    ("volume_relativo_20d", 1.002, "en", "Volume equal to normal."),
    ("volume_relativo_20d", 2.3018, "en", "Volume 2.3 times normal."),
    ("volume_relativo_20d", np.nan, "en", "Volume: no data."),
    ("retorno_5d", np.nan, "en", "Return over the last 5 sessions: no data."),
    ("distancia_media_20d", -0.068, "pt", "Preço 6,8% abaixo da média de 20 pregões."),
    ("distancia_media_20d", 0.0001, "pt", "Preço igual à média de 20 pregões."),
    ("volume_relativo_20d", 0.70, "pt", "Volume 30% abaixo do normal."),
    ("volume_relativo_20d", 1.30, "pt", "Volume 30% acima do normal."),
    ("volume_relativo_20d", 1.002, "pt", "Volume igual ao normal."),
    ("volume_relativo_20d", 2.3018, "pt", "Volume 2,3 vezes o normal."),
    ("volume_relativo_20d", np.nan, "pt", "Volume: sem dado."),
    ("retorno_5d", np.nan, "pt", "Retorno em 5 pregões: sem dado."),
])
def test_casos_limite(campo, valor, idioma, trecho):
    assert trecho in estado.gerar_estado(_linha(**{campo: valor}), idioma=idioma)


@pytest.mark.parametrize("data, esperado", [
    ("2026-01-05", "January 5, 2026"), ("2026-12-31", "December 31, 2026"),
])
def test_data_em_ingles_sem_ambiguidade(data, esperado):
    assert f"trading session of {esperado}." in estado.gerar_estado(_linha(data=pd.Timestamp(data)))


def test_mesmo_dado_gera_sempre_o_mesmo_texto(acoes):
    """Determinismo: o cache das APIs depende de o texto ser idêntico entre execuções."""
    amostra = acoes.sample(50, random_state=0)
    assert estado.gerar_estados(amostra).equals(estado.gerar_estados(amostra))
    assert estado.gerar_estados(amostra)["estado"].is_unique
