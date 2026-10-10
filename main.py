"""
main.py
Linha de produção de sinais para swing trade na B3 (etapa 1: só preço e volume).

    python main.py atualizar     # baixa B3 + Yahoo, monta o universo (precisa de internet)
    python main.py laboratorio   # testa a linha de 2016 a 2024 e escolhe a configuração
    python main.py sinais        # compras e vendas para o próximo pregão (SÓ PARA LEITURA)
    python main.py prova-final   # abre a reserva de 2025+ UMA vez e aplica os critérios
"""

import argparse
import sys

# Garante acentos corretos no terminal do Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def comando_atualizar() -> None:
    from src import atualizar
    atualizar.rodar()


def comando_laboratorio() -> None:
    from src.laboratorio import executar
    executar.rodar()


def comando_sinais() -> None:
    from src import hoje
    hoje.rodar()


def comando_prova_final() -> None:
    from src.laboratorio import prova_final
    try:
        prova_final.rodar()
    except prova_final.ErroProvaFinal as erro:
        print(f"Prova final não rodou: {erro}")


COMANDOS = {
    "atualizar": comando_atualizar,
    "laboratorio": comando_laboratorio,
    "sinais": comando_sinais,
    "prova-final": comando_prova_final,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Linha de produção de sinais (B3, swing trade)")
    parser.add_argument("comando", choices=list(COMANDOS), help="o que rodar")
    args = parser.parse_args()
    COMANDOS[args.comando]()


if __name__ == "__main__":
    main()
