"""
executar.py
Etapa "python main.py laboratorio": testa a linha de LAB_INICIO até o fim de LAB_FIM.

    1. Recorta a base: nada de RESERVA_INICIO em diante entra aqui (nem preço, nem rótulo).
    2. Para cada modelo: probabilidades com walk-forward mensal e embargo; Brier.
    3. Para cada k e p_min da grade FIXA: sinais -> carteira com custos -> métricas.
    4. Toda configuração testada vai para resultados/laboratorio/tentativas.csv.
    5. Escolha (especificação 7): menor Brier; dentro dele, maior excesso médio anual.
    6. Relatório em resultados/laboratorio/relatorio.md e escolha em escolha.json.
"""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.base import Base, carregar_base
from src.laboratorio.carteira import simular_compras
from src.laboratorio.metricas import (acerto_sinais, bateu_aleatorios, brier, prob_referencia,
                                      resumo_carteira)
from src.laboratorio.walk_forward import gerar_probabilidades, ultimo_dia_avaliavel
from src.sinais import gerar_sinais


def recortar_laboratorio(base: Base) -> Base:
    corte = pd.Timestamp(config.RESERVA_INICIO)
    calendario = base.calendario[base.calendario < corte]
    tabela = base.tabela[base.tabela["data"] < corte].copy()
    ultimo = ultimo_dia_avaliavel(calendario, calendario[-1])
    depende_da_reserva = tabela["data"] > ultimo if ultimo is not None else slice(None)
    tabela.loc[depende_da_reserva, ["excesso", "rotulo"]] = np.nan
    return Base(tabela=tabela, calendario=calendario, retornos=base.retornos[base.retornos.index < corte])


def avaliar_probabilidades(probs: pd.DataFrame, base: Base) -> dict:
    m = probs.merge(base.tabela[["data", "ticker", "rotulo"]], on=["data", "ticker"]).dropna(subset=["rotulo"])
    referencia = m["data"].map(prob_referencia(base.tabela, base.calendario))
    ok = referencia.notna()
    return {
        "brier": brier(m.loc[ok, "prob"], m.loc[ok, "rotulo"]),
        "brier_referencia": brier(referencia[ok], m.loc[ok, "rotulo"]),
        "previsoes": int(ok.sum()),
    }


def avaliar_configuracao(base: Base, probs: pd.DataFrame, tipo: str, k: int, p_min: float,
                         inicio, fim_sinais, fim_carteira, rng) -> dict:
    sinais = gerar_sinais(probs[probs["data"] <= pd.Timestamp(fim_sinais)], k, p_min)
    retorno = simular_compras(sinais, base.retornos, inicio, fim_carteira)
    retorno_bova = base.retornos[config.REFERENCIA].reindex(retorno.index).fillna(0.0)
    dias = max(probs["data"].nunique(), 1)
    return {
        "tipo": tipo, "k": k, "p_min": p_min,
        **resumo_carteira(retorno, retorno_bova),
        "acerto_compras": acerto_sinais(sinais, base.tabela, "compra"),
        "acerto_vendas": acerto_sinais(sinais, base.tabela, "venda"),
        "bateu_aleatorios": bateu_aleatorios(sinais, base.tabela, config.N_ALEATORIOS, rng),
        "compras_por_dia": float((sinais["sinal"] == "compra").sum() / dias),
    }


def escolher(brier_por_tipo: dict, linhas: list[dict]) -> dict:
    tipo = min(brier_por_tipo, key=brier_por_tipo.get)

    def nota(linha):
        valor = linha["excesso_medio_anual"]
        return -math.inf if valor is None or (isinstance(valor, float) and math.isnan(valor)) else valor

    melhor = max((l for l in linhas if l["tipo"] == tipo), key=nota)
    return {"tipo": tipo, "k": int(melhor["k"]), "p_min": float(melhor["p_min"])}


def executar_laboratorio(base: Base) -> tuple[list[dict], dict, dict]:
    base = recortar_laboratorio(base)
    inicio = pd.Timestamp(config.LAB_INICIO)
    fim_carteira = base.calendario[-1]
    fim_sinais = ultimo_dia_avaliavel(base.calendario, fim_carteira)

    linhas, resumo_probs = [], {}
    for indice, tipo in enumerate(config.TIPOS_MODELO):
        print(f"Laboratório: modelo '{tipo}' com retreino mensal (pode levar alguns minutos)...")
        probs = gerar_probabilidades(base.tabela, base.calendario, inicio, fim_sinais, tipo)
        resumo_probs[tipo] = avaliar_probabilidades(probs, base)
        for k in config.GRADE_K:
            for p_min in config.GRADE_P_MIN:
                rng = np.random.default_rng([config.SEMENTE, indice, k, int(round(p_min * 100))])
                linha = avaliar_configuracao(base, probs, tipo, k, p_min, inicio, fim_sinais, fim_carteira, rng)
                linhas.append({**linha, **resumo_probs[tipo]})
    escolha = escolher({t: r["brier"] for t, r in resumo_probs.items()}, linhas)
    return linhas, escolha, resumo_probs


def registrar_tentativa(linha: dict) -> None:
    arquivo = config.ARQUIVO_TENTATIVAS
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    nova = pd.DataFrame([{"registrado_em": datetime.now().isoformat(timespec="seconds"), **linha}])
    tabela = pd.concat([pd.read_csv(arquivo), nova], ignore_index=True) if arquivo.exists() else nova
    tabela.to_csv(arquivo, index=False, float_format="%.6f")


def _pct(valor) -> str:
    return "-" if valor is None or (isinstance(valor, float) and math.isnan(valor)) else f"{valor:+.1%}"


def _relatorio(linhas: list[dict], escolha: dict, resumo_probs: dict) -> str:
    texto = ["# Relatório do laboratório", "",
             f"Gerado em {datetime.now():%d/%m/%Y %H:%M}. Período: {config.LAB_INICIO} a {config.LAB_FIM} "
             "(a reserva de 2025+ não foi lida).", "",
             "## Probabilidades (Brier: menor é melhor)", "",
             "| modelo | Brier | Brier da taxa histórica | previsões |", "|---|---|---|---|"]
    for tipo, r in resumo_probs.items():
        texto.append(f"| {tipo} | {r['brier']:.4f} | {r['brier_referencia']:.4f} | {r['previsoes']} |")
    texto += ["", "## Carteira de compras (custo de 0,10% por lado)", "",
              "| modelo | k | p_min | retorno | BOVA11 | excesso total | excesso médio/ano | anos bons | "
              "pior queda | acerto compras | acerto vendas | bateu aleatórios | compras/dia |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in linhas:
        texto.append(
            f"| {l['tipo']} | {l['k']} | {l['p_min']:.2f} | {_pct(l['retorno_total'])} | {_pct(l['retorno_bova11'])} | "
            f"{_pct(l['excesso_total'])} | {_pct(l['excesso_medio_anual'])} | {l['fracao_anos_bons']:.0%} | "
            f"{_pct(l['pior_queda'])} | {_pct(l['acerto_compras'])} | {_pct(l['acerto_vendas'])} | "
            f"{_pct(l['bateu_aleatorios'])} | {l['compras_por_dia']:.1f} |")
    cobertura = "-"
    por_ano = []
    if config.ARQUIVO_COBERTURA.exists():
        dados = pd.read_csv(config.ARQUIVO_COBERTURA)
        cobertura = f"{dados['cobertura'].mean():.1%}"
        medias = dados.groupby(dados["mes"].astype(str).str[:4])["cobertura"].mean()
        por_ano = ["", "## Cobertura por ano (ações do universo com preço ajustado no Yahoo)", "",
                   "| ano | cobertura média |", "|---|---|"]
        por_ano += [f"| {ano} | {valor:.1%} |" for ano, valor in medias.items()]
    texto += ["", f"**Escolha:** modelo `{escolha['tipo']}`, k = {escolha['k']}, p_min = {escolha['p_min']:.2f} "
              "(menor Brier; dentro dele, maior excesso médio anual).", "",
              f"**Cobertura média do universo com preço ajustado:** {cobertura}.", *por_ano, "",
              f"Tentativas registradas até agora: ver `{config.ARQUIVO_TENTATIVAS.name}`.", "",
              "> Os sinais continuam **só para leitura** até a prova final (`python main.py prova-final`)."]
    return "\n".join(texto) + "\n"


def rodar() -> None:
    linhas, escolha, resumo_probs = executar_laboratorio(carregar_base())
    for linha in linhas:
        registrar_tentativa(linha)
    config.PASTA_LABORATORIO.mkdir(parents=True, exist_ok=True)
    config.ARQUIVO_ESCOLHA.write_text(json.dumps(
        {**escolha, "escolhido_em": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    config.ARQUIVO_RELATORIO_LAB.write_text(_relatorio(linhas, escolha, resumo_probs), encoding="utf-8")
    print(f"\nEscolha: {escolha}")
    print(f"Relatório: {config.ARQUIVO_RELATORIO_LAB.relative_to(config.RAIZ)}")
