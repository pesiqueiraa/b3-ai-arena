"""As regras da especificação que dependem só da configuração."""

import pandas as pd

import config


def test_laboratorio_termina_antes_da_reserva():
    assert pd.Timestamp(config.LAB_INICIO) < pd.Timestamp(config.LAB_FIM) < pd.Timestamp(config.RESERVA_INICIO)


def test_download_comeca_antes_do_laboratorio():
    assert pd.Timestamp(config.DATA_DOWNLOAD_INICIO) < pd.Timestamp(config.LAB_INICIO)


def test_grade_tem_no_maximo_9_combinacoes_por_modelo():
    assert len(config.GRADE_K) * len(config.GRADE_P_MIN) <= 9
    assert len(config.TIPOS_MODELO) == 2


def test_criterios_da_prova_final():
    assert config.CRITERIO_EXCESSO_MINIMO == 0.0
    assert config.CRITERIO_ALEATORIOS == 0.90
    assert config.CRITERIO_ACERTO_COMPRAS == 0.50
