"""
avaliacao.py  (vem do antigo avaliar.py)
Compara TODOS os arquivos resultados/decisoes/decisoes_*.csv contra o gabarito.
Quando decisoes_jev.csv e decisoes_llm.csv existirem, entram na comparação sozinhos.

Regra do jogo (simplificada, igual para todos os modelos):
    - No fechamento do dia D, cada modelo escolhe quais ações "comprar".
    - A carteira divide o dinheiro igualmente entre elas e fica com elas até o fechamento de D+1.
    - Se não comprar nada, fica em caixa (retorno zero nesse dia).
    - Toda troca na carteira paga um custo (corretagem, emolumentos e slippage).

Simplificações conhecidas (deixam o custo um pouco MENOR que na vida real):
    - o giro compara pesos iguais de um dia com pesos iguais do outro, ignorando que
      os pesos "andam" com os preços durante o dia;
    - a venda final, no último dia, não é cobrada.
"""

import zlib

import numpy as np
import pandas as pd

import config
from src import dados


def carregar_gabarito() -> tuple[pd.DataFrame, pd.Series]:
    return separar_gabarito(dados.carregar_indicadores())


def separar_gabarito(ind: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Das tabelas de indicadores, pega só o gabarito das ações e o do BOVA11 (a régua)."""
    gabarito = ind[["data", "ticker", "periodo", "gabarito_retorno_amanha"]].rename(
        columns={"gabarito_retorno_amanha": "gabarito"}
    )
    bova = ind[ind["ticker"] == config.REFERENCIA].set_index("data")["gabarito_retorno_amanha"]
    return gabarito[gabarito["ticker"] != config.REFERENCIA], bova


def curva_da_carteira(m: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Devolve o retorno diário da carteira com custos e sem custos."""
    com_custo, sem_custo = {}, {}
    pesos_anteriores: dict[str, float] = {}

    for dia, hoje in m.groupby("data"):
        compradas = hoje[hoje["decisao"] == "comprar"]
        n = len(compradas)
        pesos = {t: 1 / n for t in compradas["ticker"]} if n else {}

        bruto = compradas["gabarito"].mean() if n else 0.0
        # Giro = quanto da carteira mudou de um dia para o outro.
        tickers = set(pesos) | set(pesos_anteriores)
        giro = sum(abs(pesos.get(t, 0) - pesos_anteriores.get(t, 0)) for t in tickers)

        sem_custo[dia] = bruto
        com_custo[dia] = bruto - giro * config.CUSTO_POR_LADO
        pesos_anteriores = pesos

    return pd.Series(com_custo, dtype=float), pd.Series(sem_custo, dtype=float)


def acumulado(retornos: pd.Series) -> float:
    return float((1 + retornos).prod() - 1)


def drawdown_maximo(retornos: pd.Series) -> float:
    curva = (1 + retornos).cumprod()
    return float((curva / curva.cummax() - 1).min())


def teste_dos_aleatorios(m: pd.DataFrame, retorno_modelo: float, rng) -> float:
    """
    Sorteia N_ALEATORIOS agentes que, a cada dia, compram o MESMO número de ações que o
    modelo, mas escolhidas ao acaso. Devolve a fração desses agentes que o modelo superou.
    Usa retornos sem custo para medir só a habilidade de escolher ações.
    """
    resultado = np.ones(config.N_ALEATORIOS)
    houve_escolha = False  # se o modelo compra tudo (ou nada) todo dia, não há escolha a testar
    for _, hoje in m.groupby("data"):
        k = int((hoje["decisao"] == "comprar").sum())
        g = hoje["gabarito"].to_numpy()
        if k == 0:
            continue
        if k < len(g):
            houve_escolha = True
        sorteio = np.argsort(rng.random((config.N_ALEATORIOS, len(g))), axis=1)[:, :k]
        resultado *= 1 + g[sorteio].mean(axis=1)
    if not houve_escolha:
        return float("nan")
    return float(np.mean(retorno_modelo > resultado - 1))


def rng_do_modelo(nome: str):
    """
    CORREÇÃO: cada modelo tem o seu próprio sorteio, derivado da SEMENTE e do nome.
    Antes, um único sorteio era dividido entre todos os modelos em ordem alfabética,
    e incluir um modelo novo mudava o resultado dos outros.
    """
    return np.random.default_rng([config.SEMENTE, zlib.crc32(nome.encode("utf-8"))])


def faltando_na_janela(decisoes: pd.DataFrame, gabarito: pd.DataFrame,
                       esperado: pd.DataFrame | None = None) -> int:
    """
    CORREÇÃO: conta quantos pares (dia, ação) da janela limpa ficaram sem decisão.
    Antes, um modelo que pulasse dias era comparado em menos dias que os outros, sem aviso.
    'esperado' permite usar outra janela (ex.: a avaliação extra do ML em 2025).
    """
    if esperado is None:
        esperado = gabarito.loc[gabarito["periodo"] == "teste_limpo", ["data", "ticker"]]
    m = esperado.merge(decisoes[["data", "ticker"]], on=["data", "ticker"], how="left", indicator=True)
    return int((m["_merge"] == "left_only").sum())


def avaliar_modelo(nome: str, decisoes: pd.DataFrame, gabarito: pd.DataFrame,
                   bova: pd.Series, rng, esperado: pd.DataFrame | None = None) -> dict:
    m = decisoes.merge(gabarito.drop(columns="periodo"), on=["data", "ticker"], how="inner")
    m = m.dropna(subset=["gabarito"])  # o último dia ainda não tem resposta

    com_custo, sem_custo = curva_da_carteira(m)
    compras = m[m["decisao"] == "comprar"]
    # CORREÇÃO: "previu alta" agora é a própria decisão. Antes era recalculado com
    # prob_subir > 0,5 a partir do valor arredondado no CSV, o que podia discordar da decisão.
    previu_alta = m["decisao"] == "comprar"

    return {
        "modelo": nome,
        "dias": len(com_custo),
        "faltando": faltando_na_janela(decisoes, gabarito, esperado),
        "compras_por_dia": len(compras) / max(len(com_custo), 1),
        "retorno_com_custos": acumulado(com_custo),
        "retorno_sem_custos": acumulado(sem_custo),
        "retorno_bova11": acumulado(bova.reindex(com_custo.index).fillna(0)),
        "acerto_das_compras": float((compras["gabarito"] > 0).mean()) if len(compras) else float("nan"),
        "acerto_de_direcao": float((previu_alta == (m["gabarito"] > 0)).mean()) if len(m) else float("nan"),
        "pior_queda": drawdown_maximo(com_custo),
        "bateu_aleatorios": teste_dos_aleatorios(m, acumulado(sem_custo), rng),
    }


def rodar() -> None:
    gabarito, bova = carregar_gabarito()
    arquivos = sorted(config.PASTA_DECISOES.glob("decisoes_*.csv"))
    if not arquivos:
        raise SystemExit("Nenhum resultados/decisoes/decisoes_*.csv encontrado. Rode antes: python main.py ml")

    linhas = []
    for arquivo in arquivos:
        nome = arquivo.stem.replace("decisoes_", "")
        decisoes = pd.read_csv(arquivo, parse_dates=["data"])
        linhas.append(avaliar_modelo(nome, decisoes, gabarito, bova, rng_do_modelo(nome)))

    tabela = pd.DataFrame(linhas).sort_values("retorno_com_custos", ascending=False)
    config.PASTA_RESULTADOS.mkdir(exist_ok=True)
    tabela.to_csv(config.ARQUIVO_RESULTADO, index=False, float_format="%.6f")

    # Impressão amigável, em porcentagem.
    exibir = tabela.copy()
    formatos = {
        "retorno_com_custos": "{:+.1%}", "retorno_sem_custos": "{:+.1%}", "retorno_bova11": "{:+.1%}",
        "pior_queda": "{:.1%}",
        "acerto_das_compras": "{:.0%}", "acerto_de_direcao": "{:.0%}", "bateu_aleatorios": "{:.0%}",
    }
    for col, fmt in formatos.items():
        exibir[col] = exibir[col].map(lambda v, fmt=fmt: "-" if pd.isna(v) else fmt.format(v))
    exibir["compras_por_dia"] = exibir["compras_por_dia"].map(lambda v: f"{v:.1f}")

    print(exibir.to_string(index=False))
    print(f"\nTabela completa salva em {config.ARQUIVO_RESULTADO.relative_to(config.RAIZ)}")

    incompletos = tabela[tabela["faltando"] > 0]
    for _, linha in incompletos.iterrows():
        print(f"\nAtenção: '{linha['modelo']}' não decidiu {linha['faltando']} par(es) (dia, ação) "
              "da janela limpa. A comparação com os outros modelos fica desigual.")

    dias = int(tabela["dias"].max())
    print("\nComo ler:")
    print("  bateu_aleatorios = % de agentes que escolhem ações ao acaso que o modelo superou.")
    print("  Perto de 50% é o que se espera de pura sorte; só acima de ~95% começa a sugerir habilidade.")
    print("  Com poucos pregões, um único dia muito bom pode, sozinho, empurrar esse número para cima.")
    if dias < 60:
        print(f"\nAtenção: são só {dias} pregões. Com tão poucos dias, qualquer diferença ainda é")
        print("principalmente ruído. Continue rodando todo dia para acumular resultados.")
