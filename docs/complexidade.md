# Análise de Complexidade Assintótica (Big O) - Questão 1

Esta análise documenta a complexidade temporal e espacial (memória) dos três algoritmos que compõem a pipeline da Questão 1.

---

## Definição de Variáveis
* $N$: Número de SKUs no catálogo.
* $P$: Número de níveis de preço discretos permitidos por SKU.
* $G$: Número máximo de gerações do Algoritmo Genético.
* $K$: Tamanho da população no Algoritmo Genético.
* $S$: Subconjunto de SKUs selecionado para a Etapa A ($S = 6$).

---

## 1. Etapa A: Brute Force (Força Bruta)
* **Complexidade de Tempo:** $\mathcal{O}(P^S)$
  * *Explicação:* Para um subconjunto de tamanho $S$, o algoritmo testa todas as permutações possíveis de preço. Como cada produto tem $P$ opções, o espaço de busca exaustivo cresce exponencialmente.
* **Complexidade de Memória:** $\mathcal{O}(S)$
  * *Explicação:* Requer apenas armazenamento temporário para a combinação de preços em avaliação e a melhor solução encontrada até o momento.
* **Gargalo / Limite de Viabilidade:** Para $S > 10$, a complexidade exponencial $\mathcal{O}(P^S)$ inviabiliza a execução em tempo útil. Por essa razão, é aplicado exclusivamente como benchmark na instância reduzida ($S=6$).

---

## 2. Etapa B: Algoritmo Genético (GA)
* **Complexidade de Tempo:** $\mathcal{O}(G \times K \times N)$
  * *Explicação:* A cada geração ($G$), o algoritmo avalia a aptidão (*fitness*) de todos os indivíduos ($K$). O cálculo da função de aptidão percorre o catálogo completo de $N$ SKUs.
* **Complexidade de Memória:** $\mathcal{O}(K \times N)$
  * *Explicação:* O consumo espacial é dominado pelo armazenamento da população atual de $K$ indivíduos, onde cada indivíduo contém um vetor de $N$ preços.

---

## 3. Etapa C: Algoritmo Greedy (Guloso)
* **Complexidade de Tempo:** $\mathcal{O}(N \log N)$
  * *Explicação:* O tempo é dominado pela ordenação dos $N$ SKUs com base na razão de benefício marginal por real investido ($\mathcal{O}(N \log N)$), seguido por uma iteração linear $\mathcal{O}(N)$ para alocação do orçamento promocional.
* **Complexidade de Memória:** $\mathcal{O}(N)$
  * *Explicação:* Necessita de estruturas lineares auxiliares para armazenar os índices, razões calculadas e a distribuição de incentivos por SKU.