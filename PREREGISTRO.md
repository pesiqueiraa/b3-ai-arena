# Pré-registro do plano de análise

**Registrado em 09/10/2026, antes de qualquer decisão do Jev ou do LLM existir.**
A data e o conteúdo deste arquivo ficam provados pelo commit que o introduziu.
Qualquer mudança posterior precisa aparecer na seção "Desvios", com data e motivo.

Por que isto existe: com poucos pregões, quase qualquer resultado pode ser "encontrado"
se a métrica, a duração e o critério forem escolhidos depois de ver os dados. Fixar tudo
antes torna o resultado confiável, seja ele qual for.

## 1. Pergunta e hipóteses

**Pergunta:** quando os três recebem as mesmas informações, o Jev dá probabilidades melhores
que um LLM de uso geral e que um modelo estatístico clássico para "a ação vai fechar o próximo
pregão acima do fechamento de hoje?"

| | Hipótese | Tipo |
|---|---|---|
| H1 | O Jev tem Brier menor que o ML (controle) | **principal** |
| H2 | O Jev tem Brier menor que o LLM | **principal** |
| H3 | O LLM tem Brier menor que o ML | secundária |
| H4 | Cada modelo tem Brier menor que a taxa histórica de altas | secundária |

## 2. Métrica principal: Brier

- **Brier** = média de `(prob_subir − resultado)²`, em que o resultado vale 1 se a ação fechou
  o pregão seguinte acima do fechamento do dia da decisão, e 0 caso contrário.
  Menor é melhor; dizer sempre 50% dá 0,25.
- Por que o Brier e não o acerto ou o retorno: a promessa do Jev é uma **probabilidade
  calibrada**, e o Brier mede exatamente isso. O retorno da carteira depende muito de quantas
  ações cada modelo compra (um LLM que responde sempre 0,55 vira um "sempre comprar"), e o
  acerto exige amostras muito maiores (ver item 4).
- **Taxa histórica:** probabilidade de referência igual à fração de altas de todas as ações, de
  30/01/2024 até a véspera do dia da decisão (`avaliacao.prob_referencia`).

## 3. Teste estatístico e critério de vitória

- As comparações são **pareadas**: usam só os pares (dia, ação) que os dois modelos decidiram.
- As ações sobem e caem juntas, então decisões do mesmo dia não são independentes. O teste é
  feito **por dia**: para cada pregão, a diferença média de Brier (A − B); depois, a média
  dessas diferenças e um intervalo de confiança t de Student entre os dias
  (`avaliacao.comparar_brier`).
- **Comparações principais (H1, H2):** intervalo de **97,5%** (correção de Bonferroni para 2 testes).
- **Comparações secundárias (H3, H4):** intervalo de 95%, apenas descritivas.
- **Critério:** na análise final, "A melhor que B" se o intervalo inteiro ficar abaixo de zero;
  "B melhor que A" se ficar inteiro acima; caso contrário, **empate (sem diferença detectável)**.
  Um empate é um resultado válido e será relatado como tal.

## 4. Duração: 250 pregões

- **Análise final quando a janela limpa (a partir de 15/09/2026) tiver 250 pregões com
  resultado, o que deve acontecer por volta de setembro de 2027.** Não há parada antecipada
  por resultado bom ou ruim.
- Relatórios intermediários podem ser publicados, mas sempre marcados como **parciais**
  (o `python main.py avaliar` faz isso sozinho) e sem conclusões.
- **Justificativa** (calibrada com o ML em 2025, 250 pregões, 4.000 decisões):
  - O acerto diário oscila 2,4 vezes mais do que oscilaria se as 16 ações fossem independentes.
    Para detectar +3 pontos de acerto (53% contra 50%), seriam necessários ~325 pregões. Por
    isso o acerto é métrica secundária.
  - Com 250 pregões, a menor diferença de Brier detectável (80% de poder) fica em torno de
    **0,0015** para modelos com probabilidades estáveis como as do ML, e de **0,006** para
    modelos com probabilidades mais variáveis (como se espera de um LLM), ou seja, de 0,6% a
    2,4% do Brier (nas comparações principais, com 97,5%, cerca de 10% maior). Diferenças menores que isso não serão detectadas e o resultado será empate.

## 5. O que fica congelado

- **ML (controle):** os 6 indicadores de `src/modelos/ml/features.py`, a regressão logística
  de `modelo.py`, o walk-forward de `walk_forward.py` e o limiar de compra de 0,5.
  Qualquer melhoria vira **outro experimento**, com novo pré-registro.
- **Jev:** versão fixa `jev-1.13.0`.
- **Estado em texto** (`src/estado.py`) e **prompts** do Jev e do LLM: serão congelados num
  adendo a este arquivo, commitado **antes** da primeira chamada de API.
- **LLM:** o modelo (`LLM_MODELO`) será registrado no mesmo adendo. Exigência: o modelo precisa
  ter sido lançado, e ter dados de treino só até antes de 15/09/2026. Caso contrário, a janela
  limpa não é limpa para ele.
- **Custos, regras da carteira e teste dos agentes aleatórios:** como em `config.py` nesta data.

## 6. Dados faltando e decisões retroativas

- **Pares sem decisão** (falha de API, resposta inválida após a nova tentativa): ficam fora
  das comparações que envolvem aquele modelo, e o total é relatado (coluna `faltando`).
  Se um modelo deixar de decidir mais de 5% dos pares, isso será destacado nas conclusões.
- **Decisões retroativas:** decisões calculadas depois do resultado do pregão seguinte (por
  exemplo, as do ML de 15/09 a 07/10/2026, e as do Jev e do LLM nos dias anteriores à
  implementação deles) entram na análise principal, já que nenhum modelo vê o futuro por
  construção (verificado por testes).
- **Análise de sensibilidade:** a mesma comparação será repetida só com as decisões
  commitadas antes do fechamento do pregão seguinte (conferido pelo histórico do git).

## 7. Métricas secundárias (descritivas)

Acerto de direção, acerto das compras, retorno com e sem custos, retorno do BOVA11 no mesmo
período, pior queda e fração dos agentes aleatórios superada. Todas são relatadas, nenhuma
decide o resultado.

## Desvios

Nenhum até o momento.
