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

## Adendo 1: LLM (registrado em 09/10/2026, antes da primeira decisão do LLM)

### Modelo e execução
- **Modelo:** `qwen/qwen3-235b-a22b-2507` (Qwen3 235B A22B Instruct 2507), via OpenRouter.
  Pesos abertos, lançado em 21/07/2025, dados de treino até 30/06/2025 (segundo a listagem do
  OpenRouter). Cumpre a exigência do item 5: treino anterior a 15/09/2026.
- **Provedor fixo:** Alibaba, sem troca automática e exigindo suporte a todos os parâmetros
  (`provider` = `{"order": ["alibaba"], "allow_fallbacks": false, "require_parameters": true}`).
- **Parâmetros:** temperatura 0, seed 42, sem raciocínio, sem busca na web
  (sem `:online`, sem plugins e sem ferramentas).
- **Resposta inválida:** uma nova tentativa com a mensagem de correção abaixo; se falhar de
  novo, o par vai para `resultados/falhas_llm.csv` e fica fora das comparações (item 6).

### Prompt: versão `llm-prompt-v1`
Mensagem de sistema:
```
Você vai receber um resumo dos indicadores de uma ação da bolsa brasileira (B3) no fechamento de um pregão. Responda à pergunta com a probabilidade de a resposta ser "sim". Responda APENAS com um JSON no formato {"prob_subir": <número de 0 a 1>}, sem nenhum outro texto.
```
Mensagem do usuário (o estado vem de `src/estado.py`):
```
<estado>

Pergunta: A ação vai fechar o próximo pregão acima do fechamento de hoje?
```
Mensagem de correção (só na nova tentativa, depois da resposta inválida do modelo):
```
Sua resposta não seguiu o formato pedido. Responda APENAS com o JSON {"prob_subir": <número de 0 a 1>}, sem nenhum outro texto.
```

### Estado
- O experimento principal usa o estado **normal** (com ticker, nome da empresa e data), no
  formato de `src/estado.py` deste commit. Exemplo:
  > ABEV3 (Ambev), pregão de 08/10/2026. Retorno no último pregão: +1,1%. Retorno em 5 pregões:
  > +5,1%. Retorno em 20 pregões: +3,1%. Preço 3,4% acima da média de 20 pregões. Oscilação
  > diária típica nos últimos 20 pregões: 1,5%. Volume 2,3 vezes o normal.
- A versão anonimizada ("Ação A") fica reservada ao teste de memorização, que terá
  pré-registro próprio.
- O Jev receberá exatamente o mesmo estado (regra 4). A pergunta e os parâmetros do Jev
  serão registrados no Adendo 2, antes da primeira chamada do Jev.

### Como o modelo foi escolhido
Critérios: lançado antes de 15/09/2026, aceitar temperatura 0, responder sem raciocínio
obrigatório, pesos abertos (o modelo não muda nem é desativado durante o experimento) e
custo e tempo viáveis. Os candidatos foram testados com **uma única pergunta de 30/01/2024**
(fora da janela limpa), repetida. **O desempenho preditivo não foi critério**: nenhum modelo foi
avaliado na janela limpa antes deste registro.
- `deepseek/deepseek-v4-pro`: rejeitado. Sempre raciocina; a mesma pergunta deu 0,42 num
  provedor (24 s) e 0,60 em outro (10.143 tokens de raciocínio, 377 s).
- `moonshotai/kimi-k2-0905`: rejeitado. A mesma pergunta deu 0,58 e 0,45.
- `qwen/qwen3-235b-a22b-2507`: escolhido. 0,42 em todas as 5 chamadas, em 3 provedores, ~1,5 s,
  US$ 0,00004 por pergunta.

### Se algo mudar
Se o modelo ou o provedor ficarem indisponíveis, o LLM fica pausado (sem troca silenciosa).
A solução será registrada na seção "Desvios" antes de retomar.

### Decisões retroativas
As decisões do LLM de 15/09/2026 até a data da primeira execução serão calculadas
retroativamente. O modelo é de 2025 e não pode ter visto esses pregões; a análise de
sensibilidade do item 6 separa essas decisões das tomadas antes do fechamento do pregão seguinte.

## Desvios

Nenhum até o momento.
