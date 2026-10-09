"""
avaliar.py
Passo final: compara TODOS os arquivos dados/decisoes_*.csv contra o gabarito.

Quando você criar dados/decisoes_jev.csv e dados/decisoes_llm.csv (mesmo formato),
eles entram na comparação automaticamente.

Regra do jogo (simplificada, igual para todos os modelos):
    - No fechamento do dia D, cada modelo escolhe quais ações "comprar".
    - A carteira divide o dinheiro igualmente entre elas e fica com elas até o fechamento de D+1.
    - Se não comprar nada, fica em caixa (retorno zero nesse dia).
    - Toda troca na carteira paga um custo (corretagem, emolumentos e slippage).

Como rodar:
    python avaliar.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PASTA = Path("dados")
ARQUIVO_INDICADORES = PASTA / "indicadores_b3.csv"
REFERENCIA = "BOVA11"

# Custo por lado de cada operação, como fração do valor (0,0005 = 0,05%).
CUSTO_POR_LADO = 0.0005

# Quantos "macacos" (agentes aleatórios) usar no teste de sorte.
N_ALEATORIOS = 1000


def carregar_gabarito() -> tuple[pd.DataFrame, pd.Series]:
    ind = pd.read_csv(ARQUIVO_INDICADORES, parse_dates=["data"])
    gabarito = ind[["data", "ticker", "gabarito_retorno_amanha"]].rename(
        columns={"gabarito_retorno_amanha": "gabarito"}
    )
    bova = ind[ind["ticker"] == REFERENCIA].set_index("data")["gabarito_retorno_amanha"]
    return gabarito[gabarito["ticker"] != REFERENCIA], bova


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
        com_custo[dia] = bruto - giro * CUSTO_POR_LADO
        pesos_anteriores = pesos

    return pd.Series(com_custo), pd.Series(sem_custo)


def acumulado(retornos: pd.Series) -> float:
    return float((1 + retornos).prod() - 1)


def drawdown_maximo(retornos: pd.Series) -> float:
    curva = (1 + retornos).cumprod()
    return float((curva / curva.cummax() - 1).min())


def teste_dos_macacos(m: pd.DataFrame, retorno_modelo: float, rng) -> float:
    """
    Sorteia N_ALEATORIOS agentes que, a cada dia, compram o MESMO número de ações que o
    modelo, mas escolhidas ao acaso. Devolve a fração desses agentes que o modelo superou.
    Usa retornos sem custo para medir só a habilidade de escolher ações.
    """
    resultado = np.ones(N_ALEATORIOS)
    houve_escolha = False  # se o modelo compra tudo (ou nada) todo dia, não há escolha a testar
    for _, hoje in m.groupby("data"):
        k = int((hoje["decisao"] == "comprar").sum())
        g = hoje["gabarito"].to_numpy()
        if k == 0:
            continue
        if k < len(g):
            houve_escolha = True
        sorteio = np.argsort(rng.random((N_ALEATORIOS, len(g))), axis=1)[:, :k]
        resultado *= 1 + g[sorteio].mean(axis=1)
    if not houve_escolha:
        return float("nan")
    return float(np.mean(retorno_modelo > resultado - 1))


def avaliar_modelo(nome: str, decisoes: pd.DataFrame, gabarito: pd.DataFrame,
                   bova: pd.Series, rng) -> dict:
    m = decisoes.merge(gabarito, on=["data", "ticker"], how="inner")
    m = m.dropna(subset=["gabarito"])  # o último dia ainda não tem resposta

    com_custo, sem_custo = curva_da_carteira(m)
    compras = m[m["decisao"] == "comprar"]
    previu_alta = m["prob_subir"] > 0.5 if m["prob_subir"].notna().all() else m["decisao"] == "comprar"

    return {
        "modelo": nome,
        "dias": len(com_custo),
        "compras_por_dia": len(compras) / max(len(com_custo), 1),
        "retorno_com_custos": acumulado(com_custo),
        "retorno_sem_custos": acumulado(sem_custo),
        "retorno_bova11": acumulado(bova.reindex(com_custo.index).fillna(0)),
        "acerto_das_compras": float((compras["gabarito"] > 0).mean()) if len(compras) else float("nan"),
        "acerto_de_direcao": float((previu_alta == (m["gabarito"] > 0)).mean()),
        "pior_queda": drawdown_maximo(com_custo),
        "bateu_aleatorios": teste_dos_macacos(m, acumulado(sem_custo), rng),
    }


def main() -> None:
    gabarito, bova = carregar_gabarito()
    arquivos = sorted(PASTA.glob("decisoes_*.csv"))
    if not arquivos:
        raise SystemExit("Nenhum dados/decisoes_*.csv encontrado. Rode antes o ml_e_baselines.py.")

    rng = np.random.default_rng(42)
    linhas = []
    for arquivo in arquivos:
        nome = arquivo.stem.replace("decisoes_", "")
        decisoes = pd.read_csv(arquivo, parse_dates=["data"])
        linhas.append(avaliar_modelo(nome, decisoes, gabarito, bova, rng))

    tabela = pd.DataFrame(linhas).sort_values("retorno_com_custos", ascending=False)
    tabela.to_csv(PASTA / "resultado.csv", index=False, float_format="%.6f")

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
    print(f"\nTabela completa salva em {PASTA / 'resultado.csv'}")

    dias = int(tabela["dias"].max())
    print("\nComo ler:")
    print("  bateu_aleatorios = % de agentes que escolhem ações ao acaso que o modelo superou.")
    print("  Perto de 50% é o que se espera de pura sorte; só acima de ~95% começa a sugerir habilidade.")
    if dias < 60:
        print(f"\nAtenção: são só {dias} pregões. Com tão poucos dias, qualquer diferença ainda é")
        print("principalmente ruído. Continue rodando todo dia para acumular resultados.")


if __name__ == "__main__":
    main()