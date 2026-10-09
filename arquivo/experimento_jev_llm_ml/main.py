"""
main.py
Ponto de entrada do projeto. Rode no terminal, de dentro da pasta do projeto:

    python main.py baixar      # baixa preços e calcula indicadores (precisa de internet)
    python main.py ml          # decisões do ML e do baseline "sempre comprar"
    python main.py jev         # decisões do Jev
    python main.py llm         # decisões do LLM
    python main.py avaliar     # compara todos os modelos
    python main.py tudo        # todas as etapas, em ordem

    python main.py ml-historico   # EXTRA: avalia só o ML em 2025 (fora da comparação oficial)
    python main.py estado         # EXTRA: mostra exemplos do texto enviado ao Jev e ao LLM

Opções:
    --teste     (llm e jev) faz UMA pergunta num pregão antigo para testar chave e modelo.
    --refazer   recalcula TODOS os dias, apagando decisões já gravadas.
                Cuidado: decisões já commitadas no git deixam de valer como prova.
"""

import argparse
import sys

# Garante acentos corretos no terminal do Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def etapa_baixar(args) -> None:
    from src import dados
    dados.rodar()


def etapa_ml(args) -> None:
    from src.modelos import baselines, ml
    ml.rodar(refazer=args.refazer)
    baselines.rodar(refazer=args.refazer)


def etapa_jev(args) -> None:
    from src.modelos import jev
    try:
        if args.teste:
            jev.testar()
        else:
            jev.rodar(refazer=args.refazer)
    except jev.ErroDeConfiguracao as erro:
        print(f"Etapa 'jev' não rodou: {erro}")


def etapa_llm(args) -> None:
    from src.modelos import llm
    try:
        if args.teste:
            llm.testar()
        else:
            llm.rodar(refazer=args.refazer)
    except llm.ErroDeConfiguracao as erro:
        print(f"Etapa 'llm' não rodou: {erro}")


def etapa_avaliar(args) -> None:
    from src import avaliacao
    avaliacao.rodar()


def etapa_estado(args) -> None:
    from src import estado
    estado.rodar()


def etapa_ml_historico(args) -> None:
    from src.modelos.ml import historico
    historico.rodar()


ETAPAS = {
    "baixar": etapa_baixar,
    "ml": etapa_ml,
    "jev": etapa_jev,
    "llm": etapa_llm,
    "avaliar": etapa_avaliar,
}

# Etapas extras: podem ser chamadas pelo nome, mas não entram no "tudo".
EXTRAS = {
    "ml-historico": etapa_ml_historico,
    "estado": etapa_estado,
}


def rodar_etapa(nome: str, args) -> None:
    print(f"\n========== {nome.upper()} ==========")
    try:
        {**ETAPAS, **EXTRAS}[nome](args)
    except NotImplementedError as aviso:
        print(f"Etapa '{nome}' pulada: {aviso}")


def main() -> None:
    parser = argparse.ArgumentParser(description="O Pregão das IAs: Jev vs LLM vs ML na B3")
    parser.add_argument("etapa", choices=[*ETAPAS, "tudo", *EXTRAS], help="qual etapa rodar")
    parser.add_argument("--refazer", action="store_true",
                        help="recalcula todos os dias (apaga decisões já gravadas)")
    parser.add_argument("--teste", action="store_true",
                        help="(llm e jev) faz UMA pergunta num pregão antigo, para testar chave e modelo")
    args = parser.parse_args()

    if args.etapa == "tudo":
        for nome in ETAPAS:
            rodar_etapa(nome, args)
    else:
        rodar_etapa(args.etapa, args)


if __name__ == "__main__":
    main()
