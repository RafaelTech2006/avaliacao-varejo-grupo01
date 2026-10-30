# Análise de Complexidade Assintótica (Big O) - Questões 1 e 2

Esta análise documenta a complexidade temporal e espacial (memória) dos algoritmos que compõem as pipelines da Questão 1 e da Questão 2.

---

## Definição de Variáveis
* $N$: Número de SKUs no catálogo.
* $P$: Número de níveis de preço discretos permitidos por SKU.
* $G$: Número máximo de gerações do Algoritmo Genético.
* $K$: Tamanho da população no Algoritmo Genético.
* $S$: Subconjunto de SKUs selecionado para a Etapa A da Questão 1 ($S = 6$).
* $W$: Número de semanas do horizonte de planejamento da Questão 2 ($W = 12$).
* $C_{init}$: Quantidade de cenários na simulação inicial de Monte Carlo ($C_{init} = 2.000$).
* $C_{val}$: Quantidade de cenários na validação estocástica de Monte Carlo ($C_{val} = 5.000$).
* $E$: Quantidade de estados discretizados do estoque remanescente na Programação Dinâmica.

---

## Questão 1: Operação Caixa 90

### 1. Etapa A: Brute Force (Força Bruta)
* **Complexidade de Tempo:** $\mathcal{O}(P^S)$
  * *Explicação:* Para um subconjunto de tamanho $S$, o algoritmo testa todas as permutações possíveis de preço. Como cada produto tem $P$ opções, o espaço de busca exaustivo cresce exponencialmente.
* **Complexidade de Memória:** $\mathcal{O}(S)$
  * *Explicação:* Requer apenas armazenamento temporário para a combinação de preços em avaliação e a melhor solução encontrada até o momento.

### 2. Etapa B: Algoritmo Genético (GA)
* **Complexidade de Tempo:** $\mathcal{O}(G \times K \times N)$
  * *Explicação:* A cada geração ($G$), o algoritmo avalia a aptidão (*fitness*) de todos os indivíduos ($K$). O cálculo da função de aptidão percorre o catálogo completo de $N$ SKUs.
* **Complexidade de Memória:** $\mathcal{O}(K \times N)$
  * *Explicação:* O consumo espacial é dominado pelo armazenamento da população atual de $K$ indivíduos, onde cada indivíduo contém um vetor de $N$ preços.

### 3. Etapa C: Algoritmo Greedy (Guloso - Q1)
* **Complexidade de Tempo:** $\mathcal{O}(N \log N)$
  * *Explicação:* O tempo é dominado pela ordenação dos $N$ SKUs com base na razão de benefício marginal por real investido ($\mathcal{O}(N \log N)$), seguido por uma iteração linear $\mathcal{O}(N)$ para alocação do orçamento promocional.
* **Complexidade de Memória:** $\mathcal{O}(N)$
  * *Explicação:* Necessita de estruturas lineares auxiliares para armazenar os índices, razões calculadas e a distribuição de incentivos por SKU.

---

## Questão 2: Plano de Sobrevivência em 12 Semanas

### 4. Etapa A: Monte Carlo (Estimativa e Validação)
* **Complexidade de Tempo:** 
  * *Inicial ($2.000$ cenários):* $\mathcal{O}(N \times M \times C_{init})$[cite: 5]
  * *Validação Final ($5.000$ cenários):* $\mathcal{O}(N \times W \times C_{val})$[cite: 5]
  * *Explicação:* Processamento vetorizado via NumPy aplicando estocasticidade sobre a demanda, concorrentes e custos logísticos.
* **Complexidade de Memória:** $\mathcal{O}(N \times C_{val})$
  * *Explicação:* Armazenamento dos vetores de caixa simulados em paralelo para o cálculo dos percentis de risco ($P_5$).

### 5. Etapa B: Programação Dinâmica (DP)
* **Complexidade de Tempo:** $\mathcal{O}(N \times W \times E \times M)$[cite: 5]
  * *Explicação:* Para cada um dos $N$ SKUs, resolve-se o subproblema em $W$ semanas utilizando $E$ estados discretizados de estoque e $M$ opções de multiplicadores de preço.
* **Complexidade de Memória:** $\mathcal{O}(N \times W \times E)$
  * *Explicação:* Tabela de *memoization* que armazena os resultados ótimos para cada par de estados `(semana, estoque)` de forma a evitar recálculos redundantes.

### 6. Etapa C: Greedy Semanal (Q2)
* **Complexidade de Tempo:** $\mathcal{O}(W \times N \log N)$[cite: 5]
  * *Explicação:* A cada uma das $W$ semanas, os $N$ SKUs são ordenados por ordem decrescente de ROI marginal para alocação gulosa do orçamento promocional restrito.
* **Complexidade de Memória:** $\mathcal{O}(N)$
  * *Explicação:* Estrutura de lista de candidatos e registo de alocações semanais.