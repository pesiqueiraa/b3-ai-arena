"""
atualizar.py
Etapa "python main.py atualizar": baixa e prepara tudo o que a linha usa.

    1. B3 (COTAHIST) de DATA_DOWNLOAD_INICIO até hoje. O ano corrente é sempre baixado de novo.
    2. Universo de cada mês (as 100 mais líquidas, uma por empresa).
    3. Yahoo: preços ajustados de todas as ações que já passaram pelo universo, mais o BOVA11.
    4. Calendário de pregões e relatório de cobertura.

Na primeira vez são ~10 anos de dados da B3 (alguns minutos); depois, só o ano corrente.
"""

from datetime import date

import pandas as pd

import config
from src import universo as universo_mod
from src.base import montar_calendario
from src.coleta import b3, yahoo


def ano_atual() -> int:
    return date.today().year


def _baixar_b3() -> pd.DataFrame:
    partes = []
    for ano in range(int(config.DATA_DOWNLOAD_INICIO[:4]), ano_atual() + 1):
        print(f"B3 {ano}:", end=" ", flush=True)
        tabela = None
        for tentativa in (1, 2):
            try:
                caminho = b3.baixar_ano(ano, forcar=(ano == ano_atual() and tentativa == 1))
            except Exception as erro:
                caminho = b3.caminho_do_ano(ano)
                if not caminho.exists():
                    print(f"-> não consegui baixar ({erro}) e não há cópia local. Ano pulado.")
                    break
                print(f"-> não consegui baixar ({erro}); usando a cópia já baixada.", end=" ")
            try:
                tabela = b3.ler_ano(caminho)
                break
            except Exception as erro:
                if tentativa == 1:
                    print(f"-> a cópia local está ilegível ({erro}); apaguei e vou baixar de novo.", end=" ")
                    caminho.unlink(missing_ok=True)
                else:
                    print(f"-> ainda ilegível depois de baixar de novo ({erro}). Ano pulado.")
                    caminho.unlink(missing_ok=True)
        if tabela is None:
            continue
        print(f"-> {len(tabela)} linhas de ações")
        partes.append(tabela)
    if not partes:
        raise SystemExit("Nenhum arquivo da B3 disponível. Confira a internet.")
    return pd.concat(partes, ignore_index=True)


LIMITE_DE_FALHAS = 0.10   # acima disso, o download do Yahoo é considerado ruim demais


def _juntar_com_precos_anteriores(precos: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Protege o arquivo de preços de um download parcial do Yahoo. Se já existe um arquivo anterior:
    tickers que tinham dados antes e falharam agora mantêm as linhas antigas; se mais de 10% deles
    falharam, nada é sobrescrito (SystemExit).
    """
    if "ticker" not in precos.columns or precos.empty:
        raise SystemExit("O Yahoo não devolveu nenhum preço (sem internet ou bloqueio temporário?). "
                         "Nada foi alterado. Tente de novo mais tarde.")
    if config.REFERENCIA not in set(precos["ticker"]):
        raise SystemExit(f"Não consegui os preços do {config.REFERENCIA}, que é a referência. Tente de novo.")
    if not config.ARQUIVO_PRECOS.exists():
        return precos

    anteriores = pd.read_csv(config.ARQUIVO_PRECOS, parse_dates=["data"])
    tinham = set(anteriores["ticker"]) & set(tickers)
    perdidos = sorted(tinham - set(precos["ticker"]))
    if not perdidos:
        return precos
    if len(perdidos) / len(tinham) > LIMITE_DE_FALHAS:
        raise SystemExit(f"O Yahoo falhou em {len(perdidos)} de {len(tinham)} tickers que já tinham preço "
                         f"(mais de {LIMITE_DE_FALHAS:.0%}). Não sobrescrevi o arquivo de preços. "
                         "Tente de novo mais tarde.")
    print(f"Aviso: o Yahoo falhou em {len(perdidos)} ticker(s) que já tinham preço; mantive as linhas "
          f"antigas deles: {', '.join(perdidos[:20])}")
    juntos = pd.concat([precos, anteriores[anteriores["ticker"].isin(perdidos)]], ignore_index=True)
    return juntos.sort_values(["ticker", "data"]).reset_index(drop=True)


def rodar() -> None:
    for pasta in (config.PASTA_PROCESSADOS, config.PASTA_UNIVERSO, config.ARQUIVO_B3.parent,
                  config.ARQUIVO_UNIVERSO.parent, config.ARQUIVO_CALENDARIO.parent):
        pasta.mkdir(parents=True, exist_ok=True)

    tabela_b3 = _baixar_b3()
    tabela_b3.to_csv(config.ARQUIVO_B3, index=False, date_format="%Y-%m-%d")

    print("\nMontando o universo de cada mês...")
    universo = universo_mod.montar_universo(tabela_b3)
    universo.to_csv(config.ARQUIVO_UNIVERSO, index=False, date_format="%Y-%m-%d")
    troca = b3.tickers_com_troca_de_nome(tabela_b3[tabela_b3["ticker"].isin(universo["ticker"])])
    troca.to_csv(config.ARQUIVO_TROCA_DE_NOME, index=False)

    tickers = sorted(set(universo["ticker"])) + [config.REFERENCIA]
    print(f"\nBaixando preços ajustados de {len(tickers)} tickers no Yahoo...")
    precos, falharam = yahoo.baixar_varios(tickers, config.DATA_DOWNLOAD_INICIO)
    precos = _juntar_com_precos_anteriores(precos, tickers)
    precos.to_csv(config.ARQUIVO_PRECOS, index=False, date_format="%Y-%m-%d")

    calendario = montar_calendario(tabela_b3["data"], precos.loc[precos["ticker"] == config.REFERENCIA, "data"])
    pd.DataFrame({"data": calendario}).to_csv(config.ARQUIVO_CALENDARIO, index=False, date_format="%Y-%m-%d")

    cobertura = universo_mod.calcular_cobertura(universo, precos)
    cobertura.to_csv(config.ARQUIVO_COBERTURA, index=False, float_format="%.4f")
    media = cobertura["cobertura"].mean()

    print("\n===== Resumo =====")
    print(f"Meses no universo: {universo['mes'].nunique()} | ações diferentes: {universo['ticker'].nunique()}")
    print(f"Pregões: {calendario[0]:%d/%m/%Y} a {calendario[-1]:%d/%m/%Y} ({len(calendario)})")
    print(f"Sem dados no Yahoo: {len(falharam)} ticker(s)")
    print(f"Cobertura média do universo com preço ajustado: {media:.1%}")
    if media < 0.95:
        print("ATENÇÃO: cobertura abaixo de 95%. Parte do viés de sobrevivência continua "
              "(ver 'Riscos abertos' na especificação).")
    if not troca.empty:
        print(f"{len(troca)} ticker(s) com troca de nome: veja {config.ARQUIVO_TROCA_DE_NOME.name}")
