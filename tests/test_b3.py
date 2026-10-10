"""Leitura do COTAHIST (layout oficial de posições fixas da B3)."""

import zipfile

import pandas as pd
import pytest

from src.coleta import b3


def linha(tipreg="01", data="20240102", codbdi="02", ticker="PETR4", tpmerc="010",
          nome="PETROBRAS", especi="PN      N2", preult=3812, voltot=150000000) -> str:
    """Monta uma linha de 245 caracteres no layout do COTAHIST (posições começando em 1)."""
    caracteres = [" "] * 245

    def texto(inicio, fim, valor):
        tamanho = fim - inicio + 1
        caracteres[inicio - 1:fim] = list(str(valor).ljust(tamanho)[:tamanho])

    def numero(inicio, fim, valor):
        tamanho = fim - inicio + 1
        caracteres[inicio - 1:fim] = list(str(valor).zfill(tamanho))

    texto(1, 2, tipreg)
    texto(3, 10, data)
    texto(11, 12, codbdi)
    texto(13, 24, ticker)
    texto(25, 27, tpmerc)
    texto(28, 39, nome)
    texto(40, 49, especi)
    numero(109, 121, preult)
    numero(171, 188, voltot)
    return "".join(caracteres)


def test_le_os_campos_certos():
    tabela = b3.ler_linhas([linha()])
    assert tabela.to_dict("records") == [{
        "data": pd.Timestamp("2024-01-02"), "ticker": "PETR4", "nome": "PETROBRAS",
        "especie": "PN", "fechamento": 38.12, "volume_financeiro": 1500000.0,
    }]


@pytest.mark.parametrize("mudanca", [
    {"tipreg": "00"},              # cabeçalho do arquivo
    {"codbdi": "12"},              # fundo imobiliário
    {"codbdi": "14"},              # ETF
    {"tpmerc": "070"},             # opção, não é mercado à vista
    {"especi": "DRN     ED"},      # BDR
    {"especi": "CI"},              # fundo
])
def test_descarta_o_que_nao_e_acao_a_vista(mudanca):
    assert b3.ler_linhas([linha(**mudanca)]).empty


def test_aceita_units_e_preferenciais_de_classe():
    tabela = b3.ler_linhas([
        linha(ticker="TAEE11", especi="UNT     N2"),
        linha(ticker="USIM5", especi="PNA     N1"),
        linha(ticker="VALE3", especi="ON  EJ  NM"),   # marcação de ex-direito continua sendo ON
    ])
    assert list(tabela["especie"]) == ["UNT", "PNA", "ON"]


def test_linha_curta_e_ignorada():
    assert b3.ler_linhas(["01" + "x" * 50]).empty


def test_tabela_vazia_tem_as_colunas_certas():
    assert list(b3.ler_linhas([]).columns) == ["data", "ticker", "nome", "especie", "fechamento", "volume_financeiro"]


def test_le_arquivo_zip(tmp_path):
    caminho = tmp_path / "COTAHIST_A2024.ZIP"
    with zipfile.ZipFile(caminho, "w") as arquivo:
        arquivo.writestr("COTAHIST_A2024.TXT", "00COTAHIST.2024\n" + linha() + "\n99COTAHIST.2024\n")
    assert len(b3.ler_ano(caminho)) == 1


def test_detecta_ticker_com_troca_de_nome():
    tabela = pd.DataFrame({
        "ticker": ["ABCD3", "ABCD3", "EFGH3"],
        "nome": ["VELHA SA", "NOVA SA", "OUTRA"],
    })
    assert b3.tickers_com_troca_de_nome(tabela).to_dict("records") == [
        {"ticker": "ABCD3", "nomes": "NOVA SA | VELHA SA"}
    ]


class _RespostaFalsa:
    def __init__(self, conteudo: bytes):
        self.content = conteudo

    def raise_for_status(self):
        pass


def test_baixar_ano_recusa_html_e_nao_cria_arquivo(tmp_path, monkeypatch):
    import requests
    import config
    monkeypatch.setattr(config, "PASTA_B3", tmp_path)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _RespostaFalsa(b"<html>erro</html>"))
    with pytest.raises(ValueError, match="não é um ZIP"):
        b3.baixar_ano(2024)
    assert list(tmp_path.iterdir()) == []            # nem o destino, nem o .tmp


def test_baixar_ano_grava_zip_valido(tmp_path, monkeypatch):
    import io
    import requests
    import config
    monkeypatch.setattr(config, "PASTA_B3", tmp_path)
    memoria = io.BytesIO()
    with zipfile.ZipFile(memoria, "w") as z:
        z.writestr("COTAHIST_A2024.TXT", linha())
    monkeypatch.setattr(requests, "get", lambda *a, **k: _RespostaFalsa(memoria.getvalue()))
    destino = b3.baixar_ano(2024)
    assert destino.exists() and zipfile.is_zipfile(destino)
    assert [p.name for p in tmp_path.iterdir()] == [destino.name]


def test_atualizar_apaga_copia_ilegivel_e_baixa_de_novo(tmp_path, monkeypatch):
    from src import atualizar
    import config
    monkeypatch.setattr(config, "DATA_DOWNLOAD_INICIO", "2024-01-01")
    monkeypatch.setattr(atualizar, "ano_atual", lambda: 2024)
    zip_ruim = tmp_path / "2024.zip"
    zip_ruim.write_bytes(b"lixo")
    monkeypatch.setattr(atualizar.b3, "caminho_do_ano", lambda ano: zip_ruim)
    chamadas = []

    def baixar_falso(ano, forcar=False):
        chamadas.append(forcar)
        with zipfile.ZipFile(zip_ruim, "w") as z:
            z.writestr("COTAHIST_A2024.TXT", linha())
        return zip_ruim

    monkeypatch.setattr(atualizar.b3, "baixar_ano", baixar_falso)
    # primeira leitura acontece na cópia ruim: baixar_falso troca por uma boa só depois do unlink
    tabela_boa = b3.ler_linhas([linha()])
    leituras = []

    def ler_falso(caminho):
        leituras.append(1)
        if len(leituras) == 1:
            raise zipfile.BadZipFile("lixo")
        return tabela_boa

    monkeypatch.setattr(atualizar.b3, "ler_ano", ler_falso)
    resultado = atualizar._baixar_b3()
    assert len(resultado) == 1 and len(leituras) == 2


def test_atualizar_pula_ano_se_continua_ilegivel(tmp_path, monkeypatch):
    from src import atualizar
    import config
    monkeypatch.setattr(config, "DATA_DOWNLOAD_INICIO", "2023-01-01")
    monkeypatch.setattr(atualizar, "ano_atual", lambda: 2024)
    monkeypatch.setattr(atualizar.b3, "caminho_do_ano", lambda ano: tmp_path / f"{ano}.zip")

    def baixar_falso(ano, forcar=False):
        (tmp_path / f"{ano}.zip").write_bytes(b"x")
        return tmp_path / f"{ano}.zip"

    def ler_falso(caminho):
        if "2023" in caminho.name:
            raise zipfile.BadZipFile("ruim")
        return b3.ler_linhas([linha()])

    monkeypatch.setattr(atualizar.b3, "baixar_ano", baixar_falso)
    monkeypatch.setattr(atualizar.b3, "ler_ano", ler_falso)
    assert len(atualizar._baixar_b3()) == 1          # 2023 pulado, 2024 lido
