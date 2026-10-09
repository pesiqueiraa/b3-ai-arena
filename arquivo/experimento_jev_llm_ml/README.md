# Experimento encerrado: Jev vs LLM vs ML (15/09 a 09/10/2026)

Este é o primeiro formato do projeto: comparar, nas mesmas condições, três modelos decidindo todo
dia se uma ação da B3 subiria no pregão seguinte. Ele foi **encerrado em 09/10/2026 com 17
pregões e sem conclusões** (ver "Desvio 2" no `PREREGISTRO.md`).

O que fica aqui:
- `PREREGISTRO.md`: o plano de análise, os adendos e os desvios;
- `resultados/decisoes/`: as decisões de cada modelo, versionadas no git;
- `src/` e `tests/`: o código do experimento.

O código continua aqui como registro. Para rodar, entre nesta pasta, recrie a pasta `dados/`
com `python main.py baixar` e use os comandos do `main.py`.

As peças `src/modelos/jev.py`, `src/modelos/llm.py`, `src/modelos/api_comum.py` e
`src/estado.py` serão reaproveitadas na etapa 3 da linha de produção.
