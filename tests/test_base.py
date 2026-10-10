"""A base que alimenta o laboratório, os sinais e a prova final; e o comando 'atualizar'."""

import pandas as pd
import pytest

import config
from src import atualizar, base
from src.features.preco import COLUNAS_MODELO
from tests.conftest import gerar_precos, universo_completo


def test_base_tem_features_alvo_e_retornos(base_sintetica):
    t = base_sintetica.tabela
    assert list(t.columns[:2]) == ["data", "ticker"]
    assert set(COLUNAS_MODELO) <= set(t.columns) and {"excesso", "rotulo"} <= set(t.columns)
    ultimos_10 = base_sintetica.calendario[-10:]
    assert t[t["data"].isin(ultimos_10)]["rotulo"].isna().all()       # futuro ainda desconhecido
    assert t[~t["data"].isin(ultimos_10)]["rotulo"].notna().all()
    assert list(base_sintetica.retornos.index) == list(base_sintetica.calendario)


def test_acao_do_universo_sem_preco_nao_quebra():
    precos, cal = gerar_precos(n_acoes=5, fim="2016-06-30")
    u = universo_completo(precos, cal)
    u = pd.concat([u, u[u["ticker"] == "AC003"].assign(ticker="MORT3")])   # existe no universo, não no Yahoo
    b = base.montar_base(precos, u, cal)
    assert "MORT3" not in set(b.tabela["ticker"])


def test_calendario_e_a_uniao_da_b3_com_o_bova11():
    b3 = pd.to_datetime(["2024-01-02", "2024-01-03"])
    bova = pd.to_datetime(["2024-01-03", "2024-01-04"])                  # B3 do ano atrasada
    cal = base.montar_calendario(pd.Series(b3), pd.Series(bova))
    assert list(cal.strftime("%m-%d")) == ["01-02", "01-03", "01-04"]


def test_sem_atualizar_mensagem_clara(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARQUIVO_PRECOS", tmp_path / "nao_existe.csv.gz")
    with pytest.raises(SystemExit, match="atualizar"):
        base.carregar_base()


def _preparar_atualizar(tmp_path, monkeypatch, n_acoes=4):
    """Troca a internet e as pastas por dados inventados. Devolve (precos, calendario)."""
    for nome in ["PASTA_B3", "PASTA_PROCESSADOS", "PASTA_UNIVERSO"]:
        monkeypatch.setattr(config, nome, tmp_path / nome.lower())
    monkeypatch.setattr(config, "ARQUIVO_B3", tmp_path / "pp" / "b3.csv.gz")
    monkeypatch.setattr(config, "ARQUIVO_PRECOS", tmp_path / "pp" / "precos.csv.gz")
    monkeypatch.setattr(config, "ARQUIVO_CALENDARIO", tmp_path / "pp" / "calendario.csv")
    monkeypatch.setattr(config, "ARQUIVO_UNIVERSO", tmp_path / "u" / "universo.csv")
    monkeypatch.setattr(config, "ARQUIVO_COBERTURA", tmp_path / "u" / "cobertura.csv")
    monkeypatch.setattr(config, "ARQUIVO_TROCA_DE_NOME", tmp_path / "u" / "troca.csv")
    monkeypatch.setattr(config, "DATA_DOWNLOAD_INICIO", "2024-01-01")

    precos, cal = gerar_precos(n_acoes=n_acoes, inicio="2024-01-01", fim="2024-12-31")
    b3_falso = precos[precos["ticker"] != "BOVA11"].rename(columns={"fechamento_ajustado": "fechamento"})
    b3_falso = b3_falso.assign(nome="X", especie="ON", volume_financeiro=b3_falso["volume"])

    def baixar_ano_falso(ano, forcar=False):
        if ano > 2024:
            raise ConnectionError("arquivo do ano ainda não existe")
        return tmp_path / f"{ano}.zip"

    monkeypatch.setattr(atualizar.b3, "baixar_ano", baixar_ano_falso)
    monkeypatch.setattr(atualizar.b3, "caminho_do_ano", lambda ano: tmp_path / f"{ano}.zip")
    monkeypatch.setattr(atualizar.b3, "ler_ano", lambda caminho: b3_falso)
    monkeypatch.setattr(atualizar, "ano_atual", lambda: 2025)            # 2025 falha e não tem cópia: é pulado
    return precos, cal


def test_atualizar_de_ponta_a_ponta_com_fontes_falsas(tmp_path, monkeypatch):
    """Roda o 'atualizar' inteiro trocando a internet por dados inventados."""
    precos, cal = _preparar_atualizar(tmp_path, monkeypatch)
    monkeypatch.setattr(atualizar.yahoo, "baixar_varios", lambda tickers, inicio: (precos[precos["ticker"].isin(tickers)], []))

    atualizar.rodar()
    assert config.ARQUIVO_UNIVERSO.exists() and config.ARQUIVO_PRECOS.exists()
    assert len(pd.read_csv(config.ARQUIVO_CALENDARIO)) == len(cal)


def test_falha_parcial_do_yahoo_mantem_as_linhas_antigas(tmp_path, monkeypatch, capsys):
    precos, _ = _preparar_atualizar(tmp_path, monkeypatch, n_acoes=20)
    monkeypatch.setattr(atualizar.yahoo, "baixar_varios", lambda tickers, inicio: (precos[precos["ticker"].isin(tickers)], []))
    atualizar.rodar()                                                    # 1a vez: tudo bem
    antes = pd.read_csv(config.ARQUIVO_PRECOS)

    perdido = "AC00" + "3"                                               # 1 de 21 tickers (< 10%)
    parcial = precos[precos["ticker"] != perdido]
    monkeypatch.setattr(atualizar.yahoo, "baixar_varios", lambda tickers, inicio: (parcial[parcial["ticker"].isin(tickers)], [perdido]))
    atualizar.rodar()
    depois = pd.read_csv(config.ARQUIVO_PRECOS)
    assert len(depois) == len(antes)                                     # linhas antigas mantidas
    assert (depois["ticker"] == perdido).sum() == (antes["ticker"] == perdido).sum()
    assert "mantive" in capsys.readouterr().out


def test_falha_acima_de_10_por_cento_nao_sobrescreve(tmp_path, monkeypatch):
    precos, _ = _preparar_atualizar(tmp_path, monkeypatch, n_acoes=20)
    monkeypatch.setattr(atualizar.yahoo, "baixar_varios", lambda tickers, inicio: (precos[precos["ticker"].isin(tickers)], []))
    atualizar.rodar()
    antes = config.ARQUIVO_PRECOS.read_bytes()

    perdidos = [f"AC{i:02d}3" for i in range(5)]                         # 5 de 21 (> 10%)
    parcial = precos[~precos["ticker"].isin(perdidos)]
    monkeypatch.setattr(atualizar.yahoo, "baixar_varios", lambda tickers, inicio: (parcial[parcial["ticker"].isin(tickers)], perdidos))
    with pytest.raises(SystemExit, match="sobrescrevi"):
        atualizar.rodar()
    assert config.ARQUIVO_PRECOS.read_bytes() == antes


def test_yahoo_sem_nenhum_preco_da_mensagem_amigavel(tmp_path, monkeypatch):
    _preparar_atualizar(tmp_path, monkeypatch)
    monkeypatch.setattr(atualizar.yahoo, "baixar_varios",
                        lambda tickers, inicio: (pd.DataFrame(columns=["data", "ticker", "fechamento_ajustado", "volume"]), list(tickers)))
    with pytest.raises(SystemExit, match="Yahoo"):
        atualizar.rodar()
