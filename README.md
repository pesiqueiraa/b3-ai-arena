# O Pregão das IAs: Jev vs LLM vs ML na B3

> **In English (short):** a portfolio project that compares, under identical conditions, three
> kinds of models deciding every day whether to buy B3 (Brazilian stock exchange) shares:
> **Jev** (TypeSafe AI, a "System One" model that returns typed decisions with calibrated
> probabilities), a **general-purpose LLM** via API, and a **classic ML model** (walk-forward
> logistic regression). Baselines: always-buy, BOVA11 buy-and-hold and random agents.
> The focus is honest methodology: no data leakage, trading costs included, a test against
> luck, and decisions committed to git before the outcome is known.

---

## A pergunta

**Um modelo feito para devolver decisões com probabilidade calibrada (o Jev) decide melhor do que
um LLM de uso geral e do que um modelo estatístico clássico, quando os três recebem exatamente as
mesmas informações?**

A tarefa é sempre a mesma: no fechamento de cada pregão, para cada ação, responder
*"ela vai fechar o próximo pregão acima do fechamento de hoje?"* com uma probabilidade
(`prob_subir`) e uma decisão (`comprar` ou `ficar_de_fora`).

### Os competidores

| Modelo | O que é | Por que está aqui |
|---|---|---|
| **Jev** (TypeSafe AI) | Um "System One model": em vez de gerar texto, devolve uma decisão tipada com probabilidade calibrada. | É o objeto do estudo. |
| **LLM via API** | Um modelo de linguagem de uso geral, com temperatura 0 e sem busca na web. | É o que a maioria das pessoas usaria "por padrão". |
| **ML clássico** | Regressão logística retreinada a cada dia (walk-forward). | É a referência estatística barata e transparente. |
| **Sempre comprar** | Compra todas as ações todo dia. | Mede o quanto o mercado subiu sozinho. |
| **BOVA11** | Comprar e segurar o ETF do Ibovespa. | É a alternativa passiva que qualquer investidor tem. |
| **Agentes aleatórios** | 1.000 agentes que compram o mesmo número de ações que o modelo, só que sorteadas. | É o teste contra a sorte. |

## Desenho do experimento

- **Janela limpa:** só contam os pregões a partir de **15/09/2026**, data de lançamento do Jev.
  Esses dias não podiam estar no treino de nenhum dos modelos de linguagem.
- **Mesmas informações para todos:** retornos de 1, 5 e 20 pregões, distância da média de
  20 pregões, volatilidade e volume relativo. Tudo calculado **só com o passado**.
  O Jev e o LLM recebem exatamente o mesmo texto, gerado por `src/estado.py`, com os números
  já calculados e em porcentagem (o Jev é fraco em aritmética). Exemplo real:
  > ABEV3 (Ambev), pregão de 08/10/2026. Retorno no último pregão: +1,1%. Retorno em 5 pregões:
  > +5,1%. Retorno em 20 pregões: +3,1%. Preço 3,4% acima da média de 20 pregões. Oscilação
  > diária típica nos últimos 20 pregões: 1,5%. Volume 2,3 vezes o normal.

  Na versão anonimizada (para o teste de memorização), o começo vira só "Ação A.".
- **Treino walk-forward (ML):** para decidir no dia D, o modelo é treinado de novo usando só
  linhas anteriores a D. A linha de D-1 entra porque o resultado dela (de D-1 para D) já é
  conhecido no fechamento de D.
- **Carteira:** no fechamento de D, compra em pesos iguais as ações marcadas como `comprar` e
  fica com elas até o fechamento de D+1. Sem compras, fica em caixa (retorno zero).
- **Custos:** **0,05% por lado** (compra ou venda) sobre o que mudou na carteira de um dia para o
  outro. Trocar a carteira inteira custa 0,10%.
- **Métrica principal: Brier**, o erro médio da probabilidade (menor é melhor). A promessa do
  Jev é probabilidade calibrada, e é isso que o Brier mede. As comparações são pareadas por dia,
  com intervalo de confiança, e a análise final só acontece com **250 pregões** (~setembro de 2027).
  Tudo isso está fixado antes do resultado no **[PREREGISTRO.md](PREREGISTRO.md)**.
- **Teste contra a sorte:** 1.000 agentes aleatórios que, a cada dia, compram o **mesmo número**
  de ações que o modelo, só que sorteadas. A métrica `bateu_aleatorios` diz que fração deles o
  modelo superou. Perto de 50% é sorte; só acima de ~95% começa a sugerir habilidade.

### Regras que não podem ser quebradas

1. A coluna `gabarito_retorno_amanha` (o retorno do pregão seguinte) **nunca** chega a nenhum
   modelo como informação de entrada. Ela é usada para dar nota às decisões (`src/avaliacao.py`)
   e, no ML, **só** como resposta certa dos dias **anteriores** a D no treino
   (`src/modelos/ml/walk_forward.py`, o único arquivo do ML que encosta nela).
2. Para decidir no dia D, nenhum modelo usa informação posterior ao fechamento de D.
3. Todo modelo grava `resultados/decisoes/decisoes_<nome>.csv` com as colunas
   `data, ticker, prob_subir, decisao` (opcionais: `latencia_ms`, `custo_usd`).
   O formato é conferido automaticamente por `src/decisoes.py`.
4. Jev e LLM recebem exatamente o mesmo estado, gerado por `src/estado.py`.
5. Chaves de API só no `.env` (nunca no código, nunca no git).

As regras 1 e 2 são verificadas por testes automáticos (`tests/test_sem_vazamento.py`).

## Por que as decisões são versionadas no git

Os arquivos `resultados/decisoes/decisoes_*.csv` vão para o git. Cada commit tem data e hora:
**se a decisão do dia D foi commitada antes do fechamento de D+1, isso prova que ela foi tomada
sem conhecer o resultado.** Ninguém consegue "ajustar" uma decisão depois sem que o histórico do
git mostre.

Para isso funcionar, os arquivos são **só de acréscimo**: rodar de novo não recalcula dias que
já têm decisão gravada (veja `--refazer` abaixo, que deve ser evitado depois do primeiro commit).

> **Transparência:** as decisões do ML de **15/09/2026 a 07/10/2026** foram calculadas
> retroativamente em 09/10/2026, depois de os resultados existirem. Elas são uma simulação
> honesta (sem vazamento, verificada por testes), mas **não** têm a prova do git.
> A partir da decisão de **08/10/2026**, todas foram commitadas antes do pregão seguinte fechar.

## Como rodar, passo a passo

Você precisa do Python 3.12 (ou mais novo) instalado.

```bash
# 1. Entre na pasta do projeto
cd pregao-das-ias

# 2. Crie e ative um ambiente virtual (uma "caixa" só para as bibliotecas deste projeto)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Instale as bibliotecas
pip install -r requirements.txt

# 4. (Só para Jev e LLM) copie .env.example para .env e preencha as chaves

# 5. Rode os testes (não precisa de internet)
python -m pytest

# 6. Rode as etapas
python main.py baixar     # baixa preços e calcula indicadores (precisa de internet)
python main.py ml         # decisões do ML e do "sempre comprar"
python main.py avaliar    # tabela e comparações pré-registradas entre os modelos
python main.py tudo       # todas as etapas em ordem

# Extra (fora da comparação oficial): avalia só o ML em 2025 inteiro
python main.py ml-historico

# Extra: mostra exemplos do texto que o Jev e o LLM recebem
python main.py estado
```

**LLM, na primeira vez:**
1. Preencha `LLM_API_KEY` e `LLM_MODELO` no `.env`.
2. `python main.py llm --teste` faz **uma** pergunta num pregão antigo, para conferir a chave, o
   modelo e se ele responde no formato pedido. Não grava nenhuma decisão.
3. Registre o modelo e a versão do prompt (`llm-prompt-v1`) num adendo ao `PREREGISTRO.md` e
   faça o commit. Sem isso, `python main.py llm` se recusa a rodar.
4. `python main.py llm` decide a janela limpa inteira. As respostas ficam em `resultados/cache/llm/`,
   então rodar de novo não paga de novo. Respostas inválidas (mesmo após uma nova tentativa)
   ficam registradas em `resultados/falhas_llm.csv`.

**Rotina diária:** depois das ~18h (pregão fechado e consolidado), rode
`python main.py baixar`, `python main.py ml` e faça **commit e push** das decisões
**antes da abertura do próximo pregão**. Se o download vier com o dia incompleto,
o programa avisa e descarta a linha; basta rodar de novo mais tarde.

A opção `--refazer` (por exemplo, `python main.py ml --refazer`) recalcula todos os dias
e **sobrescreve** decisões já gravadas. Use só antes do primeiro commit ou para experimentos
locais, nunca no histórico oficial.

## Estrutura

```
pregao-das-ias/
├── main.py                  # CLI: python main.py baixar | ml | jev | llm | avaliar | tudo
├── config.py                # tickers, datas, custos, limiar, versões de modelo
├── src/
│   ├── dados.py             # download e indicadores
│   ├── estado.py            # linha de indicadores -> texto para Jev e LLM
│   ├── decisoes.py          # lê, valida e grava os decisoes_<modelo>.csv (só acréscimo)
│   ├── avaliacao.py         # carteira, custos, métricas e teste dos agentes aleatórios
│   └── modelos/
│       ├── ml/              # modelo de ML clássico
│       │   ├── features.py      # o que o modelo pode ver (o gabarito nunca)
│       │   ├── modelo.py        # imputação + normalização + regressão logística
│       │   ├── walk_forward.py  # retreino diário só com o passado
│       │   ├── historico.py     # avaliação extra em 2025
│       │   └── __init__.py      # a etapa "python main.py ml"
│       ├── baselines.py     # sempre_comprar
│       ├── jev.py           # Jev via API oficial da TypeSafe (versão fixa jev-1.13.0)
│       ├── api_comum.py     # cache, falhas e trava do pré-registro (Jev e LLM)
│       └── llm.py           # LLM via API compatível com OpenRouter
├── tests/
│   ├── test_sem_vazamento.py   # provas de que ninguém vê o futuro
│   ├── test_ml.py              # testes das correções da revisão
│   ├── test_avaliacao.py       # Brier, taxa histórica e comparações pré-registradas
│   ├── test_estado.py          # texto do estado: formato, anonimização, sem gabarito
│   ├── test_llm.py             # LLM com API falsa: formato, cache, nova tentativa, falhas
│   └── test_jev.py             # Jev com API falsa: formato da documentação, versão fixa, cache
├── dados/                   # CSVs baixados (fora do git)
├── resultados/
│   ├── decisoes/            # decisoes_<modelo>.csv (VERSIONADOS no git)
│   ├── cache/               # respostas das APIs (fora do git)
│   ├── extra/ml_2025/       # avaliação extra do ML em 2025 (fora da comparação oficial)
│   ├── resultado.csv        # última tabela de avaliação
│   └── comparacoes.csv      # comparações de Brier do PREREGISTRO.md
├── PREREGISTRO.md           # plano de análise fixado antes dos resultados
├── requirements.txt
├── .env.example
└── README.md
```

## Status

| Parte | Situação |
|---|---|
| Download e indicadores | pronto |
| ML (walk-forward) e "sempre comprar" | pronto e testado |
| Avaliação (Brier, comparações pareadas, custos, agentes aleatórios) | pronta e testada |
| Plano de análise (`PREREGISTRO.md`) | registrado em 09/10/2026 |
| `estado.py` (texto para Jev e LLM) | pronto e testado (formato a congelar no adendo do pré-registro) |
| Jev | pronto e testado (falta o Adendo 2 do pré-registro) |
| LLM | pronto e testado com API falsa (falta: chave, escolha do modelo e adendo do pré-registro) |

## Correções feitas na revisão dos scripts originais

Os três scripts originais estão preservados no primeiro commit do repositório. A reorganização
manteve a lógica (janela walk-forward, gabarito, regra de avaliação, custos e agentes aleatórios),
com estas correções, todas cobertas por testes em `tests/test_ml.py`:

1. **Pregão incompleto.** Quando o download acontecia antes de o Yahoo consolidar o dia, a linha
   de hoje vinha com preços de abertura, máxima e mínima zerados e volume zero. O ML descartava
   o dia em silêncio e a "decisão para o próximo pregão" usava o dia anterior. Agora essas linhas
   são removidas com aviso no download.
2. **Indicador vazio fazia o ML pular a ação.** Agora o valor vazio é preenchido com a mediana do
   treino (sem olhar o futuro), e o avaliador mostra a coluna `faltando` caso algum modelo deixe de
   decidir algum par (dia, ação) da janela limpa.
3. **Sorteio dos agentes aleatórios compartilhado entre modelos.** Incluir um modelo novo mudava o
   resultado dos outros. Agora cada modelo tem o seu próprio sorteio, fixo.
4. **Acerto de direção recalculado a partir de um valor arredondado.** Agora usa a própria coluna
   `decisao`.
5. **Decisões passaram a ser só de acréscimo** (veja "Por que as decisões são versionadas").

## Avaliação extra do ML em 2025

A janela limpa tem poucos pregões, então o ML também é avaliado em **2025 inteiro**
(250 pregões, 4.000 decisões), com o mesmo walk-forward, os mesmos custos e o mesmo teste dos
agentes aleatórios (`python main.py ml-historico`, resultados em `resultados/extra/ml_2025/`).
Jev e LLM **não** entram aqui, porque 2025 pode estar no treino deles. Resultado:

| | ML | Sempre comprar | BOVA11 |
|---|---|---|---|
| Brier (menor é melhor; taxa histórica = 0,2507) | 0,2511 | – | – |
| Acerto de direção | 49,8% (IC 95%: 48,2% a 51,3%) | 51,7% | – |
| Retorno com custos | +18,6% | +27,1% | +34,4% |
| Retorno sem custos | +36,2% | +27,1% | +34,4% |
| Superou os agentes aleatórios | 65,9% | – | – |

Conclusão: com estes indicadores, a regressão logística **não tem habilidade mensurável**
para prever a direção do dia seguinte. A acurácia é indistinguível de 50%, as probabilidades
não são melhores que a taxa histórica de altas (Brier 0,2511 contra 0,2507) e o giro diário faz os
custos consumirem metade do ganho. É exatamente o papel de um baseline honesto: mostrar quanto
um modelo "esperto" precisa superar para ser levado a sério.

## Limitações

- **Poucos pregões.** A janela limpa começou em 15/09/2026. Com algumas dezenas de dias, qualquer
  diferença entre modelos ainda é principalmente ruído.
- **Um dia pode decidir tudo.** Em 05/10/2026 (pós-1º turno das eleições), BOVA11 subiu 7,5% e
  algumas ações mais de 20%. No resultado inicial do ML, quase todo o ganho veio de ter comprado
  duas dessas ações em 02/10, uma delas com `prob_subir` de 0,501, praticamente cara ou coroa.
  Com poucos dias, até o teste dos agentes aleatórios fica sensível a um único dia assim.
- **Execução simplificada.** Supõe compra e venda exatamente no preço de fechamento, sem
  impacto de mercado. O giro ignora a variação dos pesos ao longo do dia e a venda final não é
  cobrada: o custo real seria um pouco maior.
- **Viés de sobrevivência.** A lista de ações foi escolhida hoje, entre empresas grandes que
  existem hoje. Isso favorece o "sempre comprar" no histórico.
- **Dados do Yahoo Finance.** Podem ter falhas: por exemplo, falta o BOVA11 em 24/10/2025, e
  BBAS3 e VALE3 aparecem sem volume em 26/04/2024. O preço ajustado também pode mudar levemente
  a cada novo download, quando há dividendos novos.
