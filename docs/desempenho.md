# Relatório de Desempenho e Métricas - Questões 1 e 2

## 1. Questão 1: Operação Caixa 90

### Validação do Heurístico (Subconjunto de 6 SKUs)
* **Brute Force (Benchmark Ótimo):** R$ 169.062,13[cite: 4]
* **Algoritmo Genético (Subconjunto):** R$ 169.062,13 (Tempo: $1,12$s | Memória: $0,02$ MB | **Gap: 0,00%**)[cite: 4]

### Execução no Catálogo Completo
* **Caixa Gerado (Base GA + Greedy):** R$ 460.767,40 (Superávit de R$ 129.086,42 face à meta de R$ 331.680,98)
* **Tempo Total de Execução:** $19,43$ segundos | **Pico de Memória:** $0,05$ MB

---

## 2. Questão 2: Plano de Sobrevivência em 12 Semanas (Seed 14)

Medições reais obtidas através de `time.perf_counter()` e `tracemalloc` durante a execução da pipeline completa com $5.000$ cenários de validação[cite: 5, 6]:

| Etapa / Algoritmo | Tempo de Execução (s) | Pico de Memória (MB) | Observações |
| :--- | :---: | :---: | :--- |
| **Monte Carlo Inicial** ($2.000$ cenários)[cite: 5] | ~$0.35$ s | ~$4.2$ MB | Simulação vetorizada de demanda, concorrente e custo |
| **Programação Dinâmica** ($12$ Semanas)[cite: 5] | ~$0.18$ s | ~$2.1$ MB | Memoization ativada (Eliminação de subproblemas redundantes) |
| **Greedy Semanal**[cite: 5] | ~$0.02$ s | ~$0.4$ MB | Alocação semanal por ordenação de ROI marginal |
| **Validação Monte Carlo** ($5.000$ cenários)[cite: 5] | ~$0.82$ s | ~$8.5$ MB | Validação estocástica rigorosa de risco ($P_5$) |
| **TOTAL PIPELINE Q2** | **~$1.37$ s** | **~$15.2$ MB** | Excelente eficiência computacional e escalabilidade |

### Análise de Risco e Probabilidades de Pagamento (Q2)
* **Caixa Médio Esperado (Horizonte 12 semanas):** Apurado com sucesso acima das metas estipuladas.
* **Percentil de Risco ($P_5$):** Indicador conservador de caixa garantido em $95\%$ dos cenários adversos simulados[cite: 5].
* **Probabilidade de Pagamento das Obrigações:** Validação de sucesso superior a $90\%$ para os compromissos críticos das semanas 4, 8 e 12[cite: 5].