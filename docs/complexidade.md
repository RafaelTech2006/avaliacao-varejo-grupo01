# Complexidade (Big O) — tempo e memória

## Variáveis
| Símbolo | Significado | Valor na instância |
|---|---|---|
| N | SKUs do catálogo | Q1: 30 (26 ativos) · Q2: 24 |
| L | níveis de preço permitidos | Q1: 7 · Q2: 6 |
| S | SKUs do subconjunto do Brute Force | 6 |
| P | níveis viáveis (margem ≥ piso) por SKU no subconjunto | ≤ L |
| K, G | tamanho da população e nº de gerações do GA | 50 / 100 (config A) |
| T | degraus de incentivo do Greedy | 3 |
| W | semanas | 12 |
| C_e, C_v | cenários de estimação / validação | 2.000 / 5.000 |
| E | estados de estoque da DP de um SKU (= estoque inicial + 1) | até 531 |
| B | tamanho do bloco de cenários processado por vez | 500 / 1.000 |
| M | maior estoque inicial do catálogo (colunas da pmf) | 530 |

## Questão 1
| Algoritmo | Tempo | Memória | O que domina no código | Quando deixa de ser viável → alternativa |
|---|---|---|---|---|
| Brute Force | **O(P^S · S)** | O(S) (tabelas pré-computadas + melhor combinação) | `for comb in itertools.product(...)` em `brute_force.py` | Medido: S=6 → 117.649 comb. em 1,6 s; **S=8 → 5,8 mi em 99 s**. Cresce exponencialmente → no catálogo completo (7^26) é inviável; usa-se o GA. |
| Algoritmo Genético | **O(G · K · N)** | **O(K · N)** (+ O(N·L) da tabela) | `fitness()`: indexação `tabela[ar, pop].sum(1)` da população inteira; torneio/crossover/mutação também O(K·N) | Medido até N=30.000 (2,9 s, 52 MB). Deixaria de ser viável quando K·N não couber em memória; alternativa: decompor por categoria ou reduzir K. |
| Greedy | **O(N·T·log N)** (heap) + pré-cálculo O(N·T) | O(N·T) | `heapq` + `melhor_passo()` em `greedy.py` | Medido até N=30.000 (0,9 s). Linear-logarítmico, sem limite prático. |

Pré-computar a tabela (N×L) custa O(N·L) uma vez; cada avaliação de política passa de O(N) com aritmética pandas para O(N) com indexação NumPy (≈350× mais rápido, ver `otimizacao.py`).

## Questão 2
| Algoritmo | Tempo | Memória | O que domina no código | Quando deixa de ser viável → alternativa |
|---|---|---|---|---|
| Monte Carlo (estimação) | **O(C_e · W · N · L)** | O(B·W·N + W·N·L·M) (blocos + pmf) | laço `for l, m in niveis` em `estimar()`: demanda vetorizada + `bincount` | Memória cresce com N (537 MB com N=384); alternativa: blocos menores. |
| Programação Dinâmica | **O(N · W · L · E²)** | O(N·W·E) para política (O(L·E²) temporário por semana) | `Q = recompensa + (K * V[w+1][idx_prox]).sum(-1)` em `resolver_sku()` | E² é o gargalo para estoques grandes (SKU de 530 un → 281 mil pares). Alternativa: discretizar o estoque em faixas (passo p → E/p estados). |
| Greedy semanal | **O(W · N log N)** (+ O(W·N·M) para somar a pmf) | O(N) | `np.argsort(-razao)` e `vendas_esperadas()` | Medido até N=384 (0,27 s). Sem limite prático. |
| Monte Carlo (validação) | **O(C_v · W · N)** | O(B · W · N + C_v · W) | laço semanal em `validar()` | Linear em cenários e SKUs; memória controlada por B. |

Reutilização de subproblemas (DP): `V[w+1]` é calculado uma única vez e lido por todos os (estoque, preço, vendas) da semana w. Nesta instância são **32.316 subproblemas** (Σ W·E) e 41,6 mi de transições; sem memoização seriam L^W = 6^12 ≈ 2,2 bilhões de trajetórias **por SKU**.

Tempos e memórias medidos em três tamanhos: `docs/desempenho.md`.
