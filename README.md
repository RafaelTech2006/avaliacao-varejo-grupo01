# Checkpoint 6 — Recuperação financeira de varejistas por precificação online

**Grupo: G01**

| Integrante | RM |
|---|---|
| Rafael Tavares Santos | 563487 |
| Gabriel Oliveira Alaral | 563872 |
| Felipe Yamaguchi Mesquita | 556170 |

- **Seed base:** 9 (últimos dígitos das matrículas: 7 + 2 + 0)
- **Seed final:** 1009 (1000 × 1 + 9)
- **ID da instância:** `af787c15102a`
- Python 3.12 (testado) — dependências em `requirements.txt`. Os arquivos de `data/` são os gerados pelo notebook oficial e **não foram editados**: o hash SHA-256 é conferido a cada execução (`comum/dados.py`, `tests/test_dados.py`).

## Como reproduzir

```bash
pip install -r requirements.txt
python questao_01/main.py --seed 1009 --modo completo      # Q1 (omitir --seed usa a seed_final da instância)
python questao_02/main.py --seed 1009 --cenarios 5000      # Q2
python questao_01/benchmark.py ; python questao_02/benchmark.py   # medições pequena/média/grande (Q1 leva ~2 min)
python docs/gerar_arquitetura.py                           # diagrama
python -m pytest -q                                        # 30 testes
```

Cenário-surpresa **sem alterar código** (todos por parâmetro):

```bash
python questao_01/main.py --meta-mult 1.10 --orcamento-mult 0.75 --custo-log-mult 1.15 --mutacao 0.2 --populacao 80 --remover-sku Q1-001
python questao_02/main.py --divida-mult 1.1 --concorrente-delta -0.08 --custo-log-mult 1.15 --orcamento-mult 0.75 --remover-sku Q2-001
```

## Questão 1 — Operação Caixa 90

**Nesta questão utilizamos 3 algoritmos**, encadeados:

1. **Brute Force** (`questao_01/brute_force.py`) enumera todas as combinações de preço de um subconjunto de 6 SKUs (sorteado pela seed) e produz o **ótimo exato**.
2. **Algoritmo Genético** (`genetic_algorithm.py`) roda nesse mesmo subconjunto (o ótimo do Brute Force valida o GA → *gap*) e depois no catálogo completo.
3. **Greedy** (`greedy.py`) recebe a **política de preços do GA** e distribui o orçamento de incentivos pelo maior ganho líquido marginal por real investido.

Resultados (seed 1009) e de onde vêm:

| Resultado | Valor | Arquivo / função |
|---|---|---|
| Brute Force × GA (6 SKUs, 343 combinações) | R$ 92.153,65 × R$ 92.153,65 — **gap 0,00 %** | `resultados/q1_bf_vs_ga.csv` ← `brute_force()`, `AlgoritmoGenetico.otimizar()` |
| GA catálogo completo, config A (pop 50) / B (pop 100) | R$ 466.012,29 / R$ 466.047,37 | `resultados/q1_comparacao_ga.csv` |
| GA vs ótimo exato (problema separável por SKU) | gap 0,00 % (config B) | `genetic_algorithm.otimo_separavel()` |
| Greedy: antes → depois | R$ 466.047,37 → R$ 481.154,85 (gasto R$ 20.134,06 = 22,6 % do orçamento, 15 SKUs) | `resultados/q1_antes_depois_greedy.csv` ← `GreedyIncentivos.alocar()` |
| **Meta de caixa** R$ 331.680,98 | **atingida, folga de R$ 149.473,87** | `resultados/q1_resumo.json` |
| Margem global / estoque remanescente / cobertura da dívida | 21,48 % / 1.035 un (25,0 %) / 1,56× (dívida R$ 307.568,20) | `resultados/q1_resumo.json` |
| Fitness por geração | gráfico | `resultados/q1_fitness_por_geracao.png` |
| Otimização antes/depois (fitness pandas → tabela NumPy) | 0,650 s → 0,0018 s (≈350×, 300 avaliações) | `otimizacao.py` |

Atenção: 4 SKUs (`Q1-017, Q1-024, Q1-027, Q1-029`) não têm nenhum dos 7 preços permitidos com margem ≥ piso (8,31 %); ficam fora da política e seu estoque permanece (explica parte dos 25 % remanescentes).

## Questão 2 — Plano de Sobrevivência em 12 Semanas

**Nesta questão utilizamos 3 algoritmos** (mais a revalidação):

1. **Monte Carlo** (`questao_02/monte_carlo.py`) — 2.000 cenários (demanda, preço do concorrente e custo logístico) estimam a distribuição de vendas por (semana, SKU, preço).
2. **Programação Dinâmica** (`dynamic_programming.py`) — estado (semana, estoque), decisão = nível de preço; usa as estatísticas do Monte Carlo e devolve a trajetória semanal de preços de cada SKU.
3. **Greedy** (`greedy.py`) — a cada semana concede destaques/frete promocional dentro do orçamento semanal, a partir do plano da DP.
4. **Monte Carlo de validação** — o plano completo é simulado em **5.000 cenários independentes** (seed distinta) com controle de estoque e caixa acumulado.

Resultados (seed 1009, `resultados/q2_resumo.json`, `q2_probabilidades.csv`):

| Política | P(S4) | P(S8) | P(S12) | P(todas) | Caixa final médio* |
|---|---|---|---|---|---|
| Fixa (preço 1,0) | 100 % | 100 % | 99,94 % | 99,94 % | R$ 77.769,83 |
| DP | 100 % | 100 % | 99,98 % | 99,98 % | R$ 236.724,34 |
| **DP + Greedy (plano final)** | **100 %** | **100 %** | **99,98 %** | **99,98 %** | **R$ 238.091,76** |

\* após pagar as três dívidas (R$ 54.546,73 + R$ 70.456,20 + R$ 102.275,12). Percentil de risco escolhido: **P10** (= `limite_risco_falha` da instância) = R$ 169.851,62; P5 = R$ 142.314,22. Margem média 19,9 %; estoque final médio 460 un (17,2 %); 1 cenário de falha em 5.000 (semana 12). No cenário base o risco já é baixo mesmo com preço fixo; a vantagem da DP aparece sob estresse (ex.: dívidas +30 %, concorrente −8 %, logística +15 %: P(pagar tudo) 44,3 % → 99,2 %, ver `docs/desempenho.md`).

Evidências: `docs/arquitetura.png` (fluxo), `resultados/q2_distribuicao_caixa.png`, `q2_trajetorias.png` (preço e estoque de 3 SKUs), `q2_saldo_semanal.png`, `q2_plano_precos.csv` (trajetória de todos os SKUs), `q2_greedy_destaques.csv`, `q2_benchmark.md` (3 tamanhos).

## Estrutura

```
comum/dados.py            leitura da instância + verificação SHA-256
questao_01/               modelo, brute_force, genetic_algorithm, greedy, otimizacao, benchmark, main, resultados/
questao_02/               modelo, monte_carlo, dynamic_programming, greedy, benchmark, main, resultados/
tests/                    30 testes (dados, Q1, Q2)
docs/                     arquitetura.png, complexidade.md, desempenho.md, decisoes.md
```
Documentação de uso de IA: `IA_USO.md`.
