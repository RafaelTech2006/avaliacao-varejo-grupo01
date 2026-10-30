import random
import time
import tracemalloc
import numpy as np


class AlgoritmoGenetico:
    """Implementando o Algoritmo Genético para otimização de preços sob restrições

    de margem mínima e estoque (Operação Caixa 90)[cite: 4, 11].
    """

    def __init__(
        self,
        subconjunto_skus,
        precos_permitidos,  
        precificador,
        avaliador,
        tamanho_populacao=50,
        geracoes=100,
        taxa_crossover=0.8,
        taxa_mutacao=0.1,
        tamanho_torneio=3,
        elitismo=True,
        seed=9,
    ):
        self.subconjunto_skus = subconjunto_skus
        self.precos_permitidos_dict = precos_permitidos
        self.precificador = precificador
        self.avaliador = avaliador

        # Hiperparâmetros do GA
        self.tamanho_populacao = tamanho_populacao
        self.geracoes = geracoes
        self.taxa_crossover = taxa_crossover
        self.taxa_mutacao = taxa_mutacao
        self.tamanho_torneio = tamanho_torneio
        self.elitismo = elitismo

        # Reprodutibilidade baseada em seed
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.skus_ordenados = list(subconjunto_skus["sku"])
        self.num_genes = len(self.skus_ordenados)

    def _gerar_individuo(self):
        """Um indivíduo é uma lista de preços escolhida dentro das opções permitidas por SKU."""
        individuo = []
        for sku in self.skus_ordenados:
            opcao_preco = random.choice(self.precos_permitidos_dict[sku])
            individuo.append(opcao_preco)
        return individuo

    def _gerar_populacao_inicial(self):
        """Gera a população inicial com o tamanho definido."""
        return [self._gerar_individuo() for _ in range(self.tamanho_populacao)]

    def calcular_fitness(self, individuo):
        """Avalia a aptidão do cromossomo aplicando penalização suave baseada na magnitude da violação

        de margem, permitindo a evolução gradual no catálogo completo.
        """
        caixa_total = 0.0
        penalizacao_total = 0.0

        for i, sku in enumerate(self.skus_ordenados):
            preco_proposto = individuo[i]
            row = self.subconjunto_skus.iloc[i]
            
            # Cálculo dos valores base do SKU
            var_preco = (preco_proposto - row["preco_atual"]) / row["preco_atual"]
            var_demanda = -row["elasticidade"] * var_preco
            demanda_novo = max(0, row["demanda_base_90d"] * (1 + var_demanda))
            qtd_vendida = min(demanda_novo, row["estoque"])

            receita_liquida = preco_proposto * (1 - row["taxa_marketplace"])
            custo_total_unitario = row["custo_unitario"] + row["custo_logistico"]
            caixa_unitario = receita_liquida - custo_total_unitario

            # Margem unitária proposta
            margem_unitaria = (
                caixa_unitario / preco_proposto if preco_proposto > 0 else 0
            )

            caixa_sku = qtd_vendida * caixa_unitario

            # Se violar o piso de margem mínima, aplica penalização proporcional à distância
            if margem_unitaria < self.precificador.piso_margem_minima:
                deficit_margem = self.precificador.piso_margem_minima - margem_unitaria
                penalizacao_total += deficit_margem * abs(caixa_sku) * 10.0
            else:
                caixa_total += caixa_sku

        # Fitness final: Caixa acumulado menos as penalizações por violação
        fitness_final = caixa_total - penalizacao_total
        return max(fitness_final, -1e9)

    def _selecao_torneio(self, populacao, fitnesses):
        """Seleção por Torneio estocástico/determinístico."""
        k = min(self.tamanho_torneio, len(populacao))
        competidores_idx = random.sample(range(len(populacao)), k)
        melhor_idx = max(competidores_idx, key=lambda idx: fitnesses[idx])
        return populacao[melhor_idx]

    def _crossover_ponto_unico(self, pai1, pai2):
        """Operador de Crossover de ponto único."""
        if self.num_genes < 2 or random.random() > self.taxa_crossover:
            return pai1.copy(), pai2.copy()

        ponto = random.randint(1, self.num_genes - 1)
        filho1 = pai1[:ponto] + pai2[ponto:]
        filho2 = pai2[:ponto] + pai1[ponto:]
        return filho1, filho2

    def _mutacao(self, individuo):
        """Mutação uniforme: altera o preço de um SKU para outra opção válida com probabilidade taxa_mutacao."""
        individuo_mutado = individuo.copy()
        for i, sku in enumerate(self.skus_ordenados):
            if random.random() < self.taxa_mutacao:
                opcoes = self.precos_permitidos_dict[sku]
                individuo_mutado[i] = random.choice(opcoes)
        return individuo_mutado

    def otimizar(self):
        """Executa a busca estocástica evolutiva e mede desempenho (tempo e memória)[cite: 4, 11]."""
        tracemalloc.start()
        tempo_inicio = time.perf_counter()

        populacao = self._gerar_populacao_inicial()
        historico_fitness = []

        melhor_solucao_global = None
        melhor_fitness_global = -float("inf")

        for geracao in range(self.geracoes):
            # Avaliação de Aptidão
            fitnesses = [self.calcular_fitness(ind) for ind in populacao]

            # Identificar o melhor da geração atual
            max_fit_gen = max(fitnesses)
            idx_melhor_gen = fitnesses.index(max_fit_gen)

            if max_fit_gen > melhor_fitness_global:
                melhor_fitness_global = max_fit_gen
                melhor_solucao_global = populacao[idx_melhor_gen].copy()

            historico_fitness.append(
                {
                    "geracao": geracao + 1,
                    "melhor_fitness": max_fit_gen,
                    "media_fitness": float(np.mean(fitnesses)),
                }
            )

            # Construção da Nova Geração
            nova_populacao = []

            # Preservação via Elitismo
            if self.elitismo and melhor_solucao_global is not None:
                nova_populacao.append(melhor_solucao_global.copy())

            # Reprodução (Crossover e Mutação)
            while len(nova_populacao) < self.tamanho_populacao:
                pai1 = self._selecao_torneio(populacao, fitnesses)
                pai2 = self._selecao_torneio(populacao, fitnesses)

                filho1, filho2 = self._crossover_ponto_unico(pai1, pai2)

                filho1 = self._mutacao(filho1)
                filho2 = self._mutacao(filho2)

                nova_populacao.append(filho1)
                if len(nova_populacao) < self.tamanho_populacao:
                    nova_populacao.append(filho2)

            populacao = nova_populacao

        tempo_fim = time.perf_counter()
        _, memoria_pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Recalcula estritamente o caixa real viável gerado pela melhor solução encontrada
        caixa_real_otimo = 0.0
        for i, sku in enumerate(self.skus_ordenados):
            preco = melhor_solucao_global[i]
            row = self.subconjunto_skus.iloc[i]
            
            var_preco = (preco - row["preco_atual"]) / row["preco_atual"]
            var_demanda = -row["elasticidade"] * var_preco
            demanda_novo = max(0, row["demanda_base_90d"] * (1 + var_demanda))
            qtd_vendida = min(demanda_novo, row["estoque"])
            
            receita_liquida = preco * (1 - row["taxa_marketplace"])
            custo_total_unitario = row["custo_unitario"] + row["custo_logistico"]
            caixa_unitario = receita_liquida - custo_total_unitario
            margem_unitaria = caixa_unitario / preco if preco > 0 else 0
            
            if margem_unitaria >= self.precificador.piso_margem_minima:
                caixa_real_otimo += qtd_vendida * caixa_unitario

        metricas = {
            "tempo_execucao": tempo_fim - tempo_inicio,
            "memoria_pico_mb": memoria_pico / (1024 * 1024),
            "historico": historico_fitness,
        }

        return melhor_solucao_global, caixa_real_otimo, metricas


def calcular_gap_percentual(caixa_bf, caixa_ga):
    """Calcula o Gap Percentual em relação ao benchmark ótimo do Brute Force[cite: 4, 11]."""
    if caixa_bf == -float("inf") and caixa_ga == -float("inf"):
        return 0.0  
    if caixa_bf == -float("inf") or caixa_bf == 0:
        return None
    return ((caixa_bf - caixa_ga) / abs(caixa_bf)) * 100