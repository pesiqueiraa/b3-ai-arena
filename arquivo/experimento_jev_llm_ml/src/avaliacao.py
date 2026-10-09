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


# ---------------------------------------------------------------------------
# Métrica principal: Brier (ver PREREGISTRO.md)
# ---------------------------------------------------------------------------

def prob_referencia(gabarito: pd.DataFrame) -> pd.Series:
    """
    A probabilidade "boba", por dia: a fração de altas vista no histórico até a VÉSPERA.
    No dia D, usa os gabaritos até D-1 (o de D-1 é o retorno de D-1 para D, já conhecido
    no fechamento de D). Qualquer modelo com probabilidade útil precisa bater isto.
    """
    conhecidos = gabarito.dropna(subset=["gabarito"])
    por_dia = (conhecidos["gabarito"] > 0).groupby(conhecidos["data"]).agg(["sum", "count"]).cumsum()
    taxa_ate_o_dia = por_dia["sum"] / por_dia["count"]
    # Dias sem gabarito ainda (o mais recente) repetem a última taxa conhecida;
    # depois, shift(1): no dia D, só vale o que já era conhecido até D-1.
    todos_os_dias = pd.Index(sorted(gabarito["data"].unique()))
    return taxa_ate_o_dia.reindex(todos_os_dias).ffill().shift(1)


def brier(prob: pd.Series, subiu: pd.Series) -> float:
    """Erro quadrático médio da probabilidade: 0 = perfeito, 0,25 = sempre dizer 50%. Menor é melhor."""
    return float(((prob - subiu) ** 2).mean())


def _com_resultado(decisoes: pd.DataFrame, gabarito: pd.DataFrame) -> pd.DataFrame:
    m = decisoes.merge(gabarito.drop(columns="periodo"), on=["data", "ticker"], how="inner")
    return m.dropna(subset=["gabarito"])  # o último dia ainda não tem resposta


def comparar_brier(decisoes_a: pd.DataFrame, decisoes_b: pd.DataFrame, gabarito: pd.DataFrame,
                   confianca: float = 0.95) -> dict:
    """
    Compara o Brier de A contra o de B nos MESMOS pares (dia, ação), como definido no PREREGISTRO.md.

    As ações sobem e caem juntas, então decisões do mesmo dia não são independentes.
    Por isso o teste é feito por DIA: para cada pregão, a diferença média de Brier (A - B);
    depois, a média dessas diferenças e um intervalo de confiança (t de Student) entre os dias.
    dif_media < 0 quer dizer que A errou menos que B.
    """
    from scipy import stats

    a = _com_resultado(decisoes_a, gabarito)
    b = decisoes_b[["data", "ticker", "prob_subir"]].rename(columns={"prob_subir": "prob_b"})
    m = a.merge(b, on=["data", "ticker"], how="inner")
    subiu = (m["gabarito"] > 0).astype(int)
    dif_por_dia = ((m["prob_subir"] - subiu) ** 2 - (m["prob_b"] - subiu) ** 2).groupby(m["data"]).mean()

    n = len(dif_por_dia)
    media = float(dif_por_dia.mean()) if n else float("nan")
    if n >= 2:
        margem = float(stats.t.ppf(1 - (1 - confianca) / 2, n - 1) * dif_por_dia.std(ddof=1) / np.sqrt(n))
    else:
        margem = float("nan")
    return {"dias": n, "pares": len(m), "confianca": confianca, "dif_media": media,
            "ic_baixo": media - margem, "ic_alto": media + margem}


def veredito(comparacao: dict) -> str:
    """Aplica o critério de vitória do PREREGISTRO.md a uma comparação A vs B."""
    if comparacao["dias"] < config.PREGOES_MINIMOS:
        return f"parcial ({comparacao['dias']} de {config.PREGOES_MINIMOS} pregões)"
    if comparacao["ic_alto"] < 0:
        return "A melhor que B"
    if comparacao["ic_baixo"] > 0:
        return "B melhor que A"
    return "empate (sem diferença detectável)"


# ---------------------------------------------------------------------------
# Todas as métricas de um modelo
# ---------------------------------------------------------------------------

def avaliar_modelo(nome: str, decisoes: pd.DataFrame, gabarito: pd.DataFrame,
                   bova: pd.Series, rng, esperado: pd.DataFrame | None = None) -> dict:
    m = _com_resultado(decisoes, gabarito)
    subiu = (m["gabarito"] > 0).astype(int)
    prob_ref = m["data"].map(prob_referencia(gabarito))
    brier_modelo, brier_ref = brier(m["prob_subir"], subiu), brier(prob_ref, subiu)

    com_custo, sem_custo = curva_da_carteira(m)
    compras = m[m["decisao"] == "comprar"]
    # CORREÇÃO: "previu alta" agora é a própria decisão. Antes era recalculado com
    # prob_subir > 0,5 a partir do valor arredondado no CSV, o que podia discordar da decisão.
    previu_alta = m["decisao"] == "comprar"

    return {
        "modelo": nome,
        "dias": len(com_custo),
        "faltando": faltando_na_janela(decisoes, gabarito, esperado),
        # Métrica principal. habilidade_brier > 0 = probabilidades melhores que a taxa histórica.
        "brier": brier_modelo,
        "habilidade_brier": 1 - brier_modelo / brier_ref if len(m) else float("nan"),
        "compras_por_dia": len(compras) / max(len(com_custo), 1),
        "retorno_com_custos": acumulado(com_custo),
        "retorno_sem_custos": acumulado(sem_custo),
        "retorno_bova11": acumulado(bova.reindex(com_custo.index).fillna(0)),
        "acerto_das_compras": float((compras["gabarito"] > 0).mean()) if len(compras) else float("nan"),
        "acerto_de_direcao": float((previu_alta == (m["gabarito"] > 0)).mean()) if len(m) else float("nan"),
        "pior_queda": drawdown_maximo(com_custo),
        "bateu_aleatorios": teste_dos_aleatorios(m, acumulado(sem_custo), rng),
    }


def decisoes_de_referencia(decisoes: pd.DataFrame, gabarito: pd.DataFrame) -> pd.DataFrame:
    """A probabilidade "boba" (taxa histórica de altas) nos mesmos pares de um modelo."""
    ref = decisoes[["data", "ticker"]].copy()
    ref["prob_subir"] = ref["data"].map(prob_referencia(gabarito))
    return ref


def comparacoes_preregistradas(todas: dict[str, pd.DataFrame], gabarito: pd.DataFrame) -> pd.DataFrame:
    """Roda as comparações do PREREGISTRO.md (só as dos modelos que já existem)."""
    pares = [(a, b, "principal", config.CONFIANCA_PRINCIPAL) for a, b in config.COMPARACOES_PRINCIPAIS]
    pares += [(a, b, "secundaria", config.CONFIANCA_SECUNDARIA) for a, b in config.COMPARACOES_SECUNDARIAS]
    pares += [(nome, "taxa_historica", "secundaria", config.CONFIANCA_SECUNDARIA)
              for nome in todas if nome not in config.MODELOS_SEM_PROBABILIDADE]

    linhas = []
    for a, b, tipo, confianca in pares:
        if a not in todas or (b not in todas and b != "taxa_historica"):
            continue  # modelo ainda não rodou
        dec_b = decisoes_de_referencia(todas[a], gabarito) if b == "taxa_historica" else todas[b]
        c = comparar_brier(todas[a], dec_b, gabarito, confianca)
        linhas.append({"tipo": tipo, "modelo_a": a, "modelo_b": b, **c, "veredito": veredito(c)})
    return pd.DataFrame(linhas)


def rodar() -> None:
    gabarito, bova = carregar_gabarito()
    arquivos = sorted(config.PASTA_DECISOES.glob("decisoes_*.csv"))
    if not arquivos:
        raise SystemExit("Nenhum resultados/decisoes/decisoes_*.csv encontrado. Rode antes: python main.py ml")

    todas, linhas = {}, []
    for arquivo in arquivos:
        nome = arquivo.stem.replace("decisoes_", "")
        todas[nome] = pd.read_csv(arquivo, parse_dates=["data"])
        linhas.append(avaliar_modelo(nome, todas[nome], gabarito, bova, rng_do_modelo(nome)))

    # Ordem pela métrica principal (Brier: menor é melhor).
    tabela = pd.DataFrame(linhas).sort_values("brier")
    config.PASTA_RESULTADOS.mkdir(exist_ok=True)
    tabela.to_csv(config.ARQUIVO_RESULTADO, index=False, float_format="%.6f")
    comparacoes = comparacoes_preregistradas(todas, gabarito)
    comparacoes.to_csv(config.ARQUIVO_COMPARACOES, index=False, float_format="%.6f")

    dias = int(tabela["dias"].max())
    situacao = "FINAL" if dias >= config.PREGOES_MINIMOS else "PARCIAL (não conclusivo)"
    print(f"Pregões com resultado: {dias} de {config.PREGOES_MINIMOS} previstos no PREREGISTRO.md "
          f"-> resultado {situacao}\n")

    # Impressão amigável, em porcentagem.
    exibir = tabela.copy()
    formatos = {
        "brier": "{:.4f}", "habilidade_brier": "{:+.2%}",
        "retorno_com_custos": "{:+.1%}", "retorno_sem_custos": "{:+.1%}", "retorno_bova11": "{:+.1%}",
        "pior_queda": "{:.1%}",
        "acerto_das_compras": "{:.0%}", "acerto_de_direcao": "{:.0%}", "bateu_aleatorios": "{:.0%}",
    }
    for col, fmt in formatos.items():
        exibir[col] = exibir[col].map(lambda v, fmt=fmt: "-" if pd.isna(v) else fmt.format(v))
    exibir["compras_por_dia"] = exibir["compras_por_dia"].map(lambda v: f"{v:.1f}")

    print("MÉTRICA PRINCIPAL: brier (menor é melhor) | habilidade_brier > 0 = melhor que a taxa histórica")
    print(exibir.to_string(index=False))

    if not comparacoes.empty:
        print("\nComparações pré-registradas (diferença de Brier A - B por dia; negativo = A melhor):")
        mostrar = comparacoes.copy()
        mostrar["dif_media"] = mostrar["dif_media"].map("{:+.4f}".format)
        mostrar["intervalo"] = [f"{c:.1%}: [{b:+.4f}, {a:+.4f}]"
                                for c, b, a in zip(comparacoes["confianca"], comparacoes["ic_baixo"], comparacoes["ic_alto"])]
        print(mostrar[["tipo", "modelo_a", "modelo_b", "dias", "dif_media", "intervalo", "veredito"]].to_string(index=False))

    print(f"\nTabelas salvas em {config.ARQUIVO_RESULTADO.relative_to(config.RAIZ)} "
          f"e {config.ARQUIVO_COMPARACOES.relative_to(config.RAIZ)}")

    incompletos = tabela[tabela["faltando"] > 0]
    for _, linha in incompletos.iterrows():
        print(f"\nAtenção: '{linha['modelo']}' não decidiu {linha['faltando']} par(es) (dia, ação) "
              "da janela limpa. As comparações usam só os pares que os dois modelos decidiram.")

    print("\nComo ler:")
    print("  brier = erro médio da probabilidade. 0,25 é o que dá dizer sempre 50%.")
    print("  O 'sempre_comprar' diz 100% em tudo, então o Brier dele é ruim de propósito: não é um")
    print("  modelo de probabilidade, e por isso fica fora das comparações.")
    print("  bateu_aleatorios = % de agentes que escolhem ações ao acaso que o modelo superou.")
    print("  Perto de 50% é o que se espera de pura sorte; só acima de ~95% começa a sugerir habilidade.")
    if situacao != "FINAL":
        print(f"\nAtenção: com {dias} pregões, nenhuma diferença é conclusiva. O PREREGISTRO.md fixa a")
        print(f"análise final em {config.PREGOES_MINIMOS} pregões. Até lá, os números são só acompanhamento.")
