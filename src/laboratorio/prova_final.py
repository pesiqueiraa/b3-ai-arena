"""
prova_final.py
Etapa "python main.py prova-final": abre a reserva (RESERVA_INICIO em diante) UMA VEZ, com a
configuração escolhida pelo laboratório, e aplica os critérios definidos ANTES de qualquer resultado:

    1. excesso líquido total da carteira de compras sobre o BOVA11 > 0
    2. superar pelo menos 90% dos agentes aleatórios
    3. acerto das compras > 50%

Se resultados/prova_final.json já existir, a etapa se recusa a rodar. Rodar de novo exige
apagar o arquivo, e isso fica visível no git.
"""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.base import Base, carregar_base
from src.laboratorio.executar import avaliar_configuracao, avaliar_probabilidades
from src.laboratorio.walk_forward import gerar_probabilidades, ultimo_dia_avaliavel


class ErroProvaFinal(Exception):
    """A prova final não pode rodar agora (já rodou, falta o laboratório, ou faltam pregões)."""


def _numero(valor):
    """NaN vira None no JSON (e conta como critério NÃO cumprido)."""
    return None if isinstance(valor, float) and math.isnan(valor) else valor


def criterios(linha: dict) -> dict[str, bool]:
    def maior(chave, limite, igual=False):
        valor = linha.get(chave)
        if valor is None or (isinstance(valor, float) and math.isnan(valor)):
            return False
        return valor >= limite if igual else valor > limite

    return {
        "excesso_total_positivo": maior("excesso_total", config.CRITERIO_EXCESSO_MINIMO),
        "bateu_aleatorios": maior("bateu_aleatorios", config.CRITERIO_ALEATORIOS, igual=True),
        "acerto_compras": maior("acerto_compras", config.CRITERIO_ACERTO_COMPRAS),
    }


def rodar(base: Base | None = None) -> dict:
    if config.ARQUIVO_PROVA_FINAL.exists():
        raise ErroProvaFinal(f"a prova final já foi rodada ({config.ARQUIVO_PROVA_FINAL.name} existe). "
                             "Rodar de novo exige apagar o arquivo, e isso fica registrado no git.")
    if not config.ARQUIVO_ESCOLHA.exists():
        raise ErroProvaFinal("não há escolha do laboratório. Rode antes: python main.py laboratorio")
    escolha = json.loads(config.ARQUIVO_ESCOLHA.read_text(encoding="utf-8"))
    base = base or carregar_base()

    inicio_pedido = pd.Timestamp(config.RESERVA_INICIO)
    na_reserva = base.calendario[base.calendario >= inicio_pedido]
    fim_carteira = base.calendario[-1]
    fim_sinais = ultimo_dia_avaliavel(base.calendario, fim_carteira)
    if len(na_reserva) == 0 or fim_sinais is None or fim_sinais < na_reserva[0]:
        raise ErroProvaFinal("ainda não há pregões suficientes na reserva para avaliar.")
    inicio = na_reserva[0]

    print(f"Prova final: modelo '{escolha['tipo']}', k = {escolha['k']}, p_min = {escolha['p_min']} "
          f"| reserva {inicio:%d/%m/%Y} a {fim_sinais:%d/%m/%Y}")
    probs = gerar_probabilidades(base.tabela, base.calendario, inicio, fim_sinais, escolha["tipo"])
    rng = np.random.default_rng(config.SEMENTE)
    linha = {**avaliar_configuracao(base, probs, escolha["tipo"], escolha["k"], escolha["p_min"],
                                    inicio, fim_sinais, fim_carteira, rng),
             **avaliar_probabilidades(probs, base)}
    resultado_criterios = criterios(linha)
    resultado = {
        "rodado_em": datetime.now().isoformat(timespec="seconds"),
        "periodo": [f"{inicio:%Y-%m-%d}", f"{fim_sinais:%Y-%m-%d}"],
        "escolha": escolha,
        "metricas": {chave: _numero(valor) for chave, valor in linha.items()},
        "criterios": resultado_criterios,
        "aprovado": all(resultado_criterios.values()),
    }
    # Monta os dois textos antes de gravar qualquer coisa. O JSON é o que tranca novas execuções,
    # então ele é gravado por ÚLTIMO: se o .md falhar, a prova ainda não foi "gasta".
    texto_json = json.dumps(resultado, indent=2, ensure_ascii=False)
    texto_relatorio = _relatorio(resultado)
    config.ARQUIVO_PROVA_FINAL.parent.mkdir(parents=True, exist_ok=True)
    config.ARQUIVO_RELATORIO_PROVA.write_text(texto_relatorio, encoding="utf-8")
    config.ARQUIVO_PROVA_FINAL.write_text(texto_json, encoding="utf-8")

    veredito = "APROVADA para uso com cautela" if resultado["aprovado"] else "NÃO aprovada"
    print(f"\nResultado: {veredito}")
    for nome, ok in resultado_criterios.items():
        print(f"  {'OK ' if ok else 'NÃO'}  {nome}")
    return resultado


def _relatorio(r: dict) -> str:
    m = r["metricas"]
    pct = lambda v: "-" if v is None else f"{v:+.1%}"
    linhas = [
        "# Prova final (reserva)", "",
        f"Rodada em {r['rodado_em']}. Período: {r['periodo'][0]} a {r['periodo'][1]}.",
        f"Configuração: modelo `{r['escolha']['tipo']}`, k = {r['escolha']['k']}, p_min = {r['escolha']['p_min']}.", "",
        f"- Excesso líquido total sobre o BOVA11: {pct(m.get('excesso_total'))} "
        f"({'OK' if r['criterios']['excesso_total_positivo'] else 'não'})",
        f"- Agentes aleatórios superados: {pct(m.get('bateu_aleatorios'))} "
        f"({'OK' if r['criterios']['bateu_aleatorios'] else 'não'})",
        f"- Acerto das compras: {pct(m.get('acerto_compras'))} ({'OK' if r['criterios']['acerto_compras'] else 'não'})",
        f"- Brier: {m.get('brier')} (taxa histórica: {m.get('brier_referencia')})", "",
        f"**Resultado: {'APROVADA para uso com cautela' if r['aprovado'] else 'NÃO aprovada'}.**", "",
        "> A reserva tem poucos meses: um resultado positivo é um sinal, não uma prova definitiva.",
    ]
    return "\n".join(linhas) + "\n"
