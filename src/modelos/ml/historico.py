"""
historico.py
Avaliação EXTRA do ML num período longo (por padrão, 2025 inteiro).

Por que existe: a janela limpa (desde 15/09/2026) ainda tem poucos pregões, e um único
dia pode decidir o resultado. Como o walk-forward impede o ML de ver o futuro, dá para
avaliá-lo também num período bem maior e descobrir se a acurácia dele é a real ou ruído.

O que NÃO é: não faz parte da comparação oficial com Jev e LLM (esses podem ter visto
2025 no treino). Por isso grava em resultados/extra/ml_2025/, e não em resultados/decisoes/.

Rodar:  python main.py ml-historico
"""

import numpy as np
import pandas as pd

import config
from src import avaliacao, dados
from src.modelos.baselines import sempre_comprar
from src.modelos.ml.walk_forward import gerar_decisoes


def intervalo_95(acertos: int, total: int) -> tuple[float, float]:
    """Intervalo de 95% para uma taxa de acerto (aproximação normal)."""
    p = acertos / total
    margem = 1.96 * np.sqrt(p * (1 - p) / total)
    return p - margem, p + margem


def brier(prob: pd.Series, subiu: pd.Series) -> float:
    """Erro quadrático médio da probabilidade (0 = perfeito; menor é melhor)."""
    return float(((prob - subiu) ** 2).mean())


def rodar() -> None:
    df = dados.carregar_indicadores()
    df = df[df["ticker"] != config.REFERENCIA].copy()

    inicio, fim = pd.Timestamp(config.ML_HISTORICO_INICIO), pd.Timestamp(config.ML_HISTORICO_FIM)
    dias = sorted(df.loc[(df["data"] >= inicio) & (df["data"] <= fim), "data"].unique())
    if not dias:
        raise SystemExit("Nenhum pregão no período pedido. Confira ML_HISTORICO_INICIO/FIM no config.py.")

    print(f"ML (avaliação extra): walk-forward em {len(dias)} pregões, "
          f"de {dias[0]:%d/%m/%Y} a {dias[-1]:%d/%m/%Y}. Isso leva alguns segundos...")
    decisoes_ml = gerar_decisoes(df, dias)
    decisoes_sc = sempre_comprar(df[df["data"].isin(dias)])

    config.PASTA_EXTRA_ML.mkdir(parents=True, exist_ok=True)
    decisoes_ml.to_csv(config.PASTA_EXTRA_ML / "decisoes_ml.csv", index=False,
                       date_format="%Y-%m-%d", float_format="%.6f")

    # --- Mesmas métricas da avaliação oficial, no período longo -------------
    gabarito, bova = avaliacao.carregar_gabarito()
    esperado = gabarito.loc[gabarito["data"].isin(dias), ["data", "ticker"]]
    linhas = [
        avaliacao.avaliar_modelo(nome, d, gabarito, bova, avaliacao.rng_do_modelo(nome), esperado)
        for nome, d in [("ml", decisoes_ml), ("sempre_comprar", decisoes_sc)]
    ]
    tabela = pd.DataFrame(linhas)

    # --- Métricas extras: a acurácia é real ou ruído? ------------------------
    m = decisoes_ml.merge(gabarito.drop(columns="periodo"), on=["data", "ticker"]).dropna(subset=["gabarito"])
    subiu = (m["gabarito"] > 0).astype(int)
    acertou = (m["decisao"] == "comprar") == (subiu == 1)
    baixo, alto = intervalo_95(int(acertou.sum()), len(m))
    taxa_base = subiu.mean()

    # Probabilidade "boba": sempre a fração de altas vista no treino até a véspera.
    # Se o ML não tiver Brier menor que isto, as probabilidades dele não acrescentam nada.
    altas_dia = df.dropna(subset=["gabarito_retorno_amanha"]).assign(
        s=lambda x: (x["gabarito_retorno_amanha"] > 0).astype(int)
    ).groupby("data")["s"].agg(["sum", "count"]).cumsum()
    base_ate_vespera = (altas_dia["sum"] / altas_dia["count"]).shift(1)
    prob_boba = m["data"].map(base_ate_vespera)

    # Robustez: quanto do retorno sobra sem o melhor dia?
    com_custo, _ = avaliacao.curva_da_carteira(m)
    sem_melhor_dia = avaliacao.acumulado(com_custo.drop(com_custo.idxmax()))

    por_mes = m.assign(acertou=acertou, subiu=subiu, mes=m["data"].dt.strftime("%Y-%m")).groupby("mes").agg(
        decisoes=("acertou", "size"),
        acerto_ml=("acertou", "mean"),
        fracao_que_subiu=("subiu", "mean"),
        compras_por_dia=("decisao", lambda s: (s == "comprar").sum()),
    )
    por_mes["compras_por_dia"] /= m.groupby(m["data"].dt.strftime("%Y-%m"))["data"].nunique()

    tabela.to_csv(config.PASTA_EXTRA_ML / "resumo.csv", index=False, float_format="%.6f")
    por_mes.to_csv(config.PASTA_EXTRA_ML / "por_mes.csv", float_format="%.4f")

    # --- Impressão -----------------------------------------------------------
    exibir = tabela.drop(columns="faltando").copy()
    for col in ["retorno_com_custos", "retorno_sem_custos", "retorno_bova11"]:
        exibir[col] = exibir[col].map("{:+.1%}".format)
    for col in ["acerto_das_compras", "acerto_de_direcao", "bateu_aleatorios"]:
        exibir[col] = exibir[col].map(lambda v: "-" if pd.isna(v) else f"{v:.1%}")
    exibir["pior_queda"] = exibir["pior_queda"].map("{:.1%}".format)
    exibir["compras_por_dia"] = exibir["compras_por_dia"].map("{:.1f}".format)
    print("\n" + exibir.to_string(index=False))

    print(f"\nAcerto de direção do ML: {acertou.mean():.1%} em {len(m)} decisões "
          f"(intervalo de 95%: {baixo:.1%} a {alto:.1%})")
    print(f"Fração de ações que subiram no dia seguinte (quem 'compra tudo' acerta isso): {taxa_base:.1%}")
    print(f"Brier do ML: {brier(m['prob_subir'], subiu):.4f}  |  "
          f"Brier da probabilidade 'boba' (taxa histórica de altas): {brier(prob_boba, subiu):.4f}  "
          "(menor é melhor)")
    print(f"Retorno do ML com custos sem o seu melhor dia ({com_custo.idxmax():%d/%m/%Y}): {sem_melhor_dia:+.1%}")

    print("\nPor mês:")
    mostrar = por_mes.copy()
    for col in ["acerto_ml", "fracao_que_subiu"]:
        mostrar[col] = mostrar[col].map("{:.0%}".format)
    mostrar["compras_por_dia"] = mostrar["compras_por_dia"].map("{:.1f}".format)
    print(mostrar.to_string())

    print(f"\nArquivos salvos em {config.PASTA_EXTRA_ML.relative_to(config.RAIZ)}")
    print("Lembrete: esta avaliação é só do ML e NÃO entra na comparação oficial com Jev e LLM.")
