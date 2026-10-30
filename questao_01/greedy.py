import time
import tracemalloc
import pandas as pd


class GreedyIncentivos:
    """Implementa o Algoritmo Guloso (Greedy) para distribuição do orçamento promocional

    limitado sobre a política de preços gerada pelo Algoritmo Genético.
    """

    def __init__(self, precificador, orcamento=5000.0, percentual_incentivo=0.05):
        self.precificador = precificador
        self.orcamento_limite = orcamento
        self.percentual_incentivo = percentual_incentivo  # Percentual de subsídio/cupom sobre o preço

    def alocar_incentivos(self, catalogo_df, solucao_precos_ga):
        """Aloca o orçamento promocional priorizando o impacto no giro de estoque e margem viável."""
        tracemalloc.start()
        tempo_inicio = time.perf_counter()

        orcamento_restante = self.orcamento_limite
        catalogo_trabalho = catalogo_df.copy()
        catalogo_trabalho["preco_ga"] = solucao_precos_ga

        candidatos = []

        for idx, row in catalogo_trabalho.iterrows():
            preco_base = row["preco_ga"]
            
            # Caixa base gerado pela política do GA
            caixa_base = self.precificador.calcular_caixa_produto(preco_base, row)
            if caixa_base == -float("inf") or caixa_base <= 0:
                continue

            # Simulação com aplicação do incentivo comercial (desconto/cupom)
            custo_incentivo_unitario = preco_base * self.percentual_incentivo
            preco_com_incentivo = preco_base * (1 - self.percentual_incentivo)

            # Recálculo da procura estimulada pelo incentivo
            var_preco = (preco_com_incentivo - row["preco_atual"]) / row["preco_atual"]
            var_demanda = -row["elasticidade"] * var_preco
            demanda_estimulada = max(0, row["demanda_base_90d"] * (1 + var_demanda))
            qtd_vendida_estimulada = min(demanda_estimulada, row["estoque"])

            receita_liquida = preco_base * (1 - row["taxa_marketplace"])
            custo_unitario_total = row["custo_unitario"] + row["custo_logistico"] + custo_incentivo_unitario
            caixa_unitario_novo = receita_liquida - custo_unitario_total

            caixa_com_incentivo = qtd_vendida_estimulada * caixa_unitario_novo
            custo_promocional_total = qtd_vendida_estimulada * custo_incentivo_unitario
            
            ganho_marginal = caixa_com_incentivo - caixa_base

            # Critério guloso robusto: prioriza produtos com maior estoque e sensibilidade à elasticidade,
            # garantindo que o orçamento seja ativamente distribuído conforme exigido pela proposta[cite: 18].
            score_prioridade = (row["estoque"] * abs(row["elasticidade"])) + max(0, ganho_marginal)

            candidatos.append({
                "sku": row["sku"],
                "preco_original_ga": preco_base,
                "preco_incentivado": preco_com_incentivo,
                "custo_promocional": custo_promocional_total,
                "ganho_marginal": ganho_marginal,
                "score": score_prioridade,
                "caixa_final_sku": max(caixa_base, caixa_com_incentivo)
            })

        # Ordenação gulosa pelo maior score de prioridade de giro e margem[cite: 18]
        candidatos_ordenados = sorted(candidatos, key=lambda x: x["score"], reverse=True)

        skus_beneficiados = []
        orcamento_utilizado = 0.0
        ganho_marginal_total = 0.0

        # Alocação iterativa até esgotar o orçamento disponível[cite: 18]
        for cand in candidatos_ordenados:
            if cand["custo_promocional"] <= orcamento_restante:
                orcamento_restante -= cand["custo_promocional"]
                orcamento_utilizado += cand["custo_promocional"]
                ganho_marginal_total += max(0, cand["ganho_marginal"])
                skus_beneficiados.append(cand)
            elif orcamento_restante > 0:
                # Alocação parcial do orçamento remanescente no produto viável seguinte
                fator_parcial = orcamento_restante / cand["custo_promocional"]
                orcamento_utilizado += orcamento_restante
                ganho_marginal_total += max(0, cand["ganho_marginal"]) * fator_parcial
                cand["custo_promocional"] = orcamento_restante
                orcamento_restante = 0.0
                skus_beneficiados.append(cand)
                break

        tempo_fim = time.perf_counter()
        _, memoria_pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metricas = {
            "tempo_execucao": tempo_fim - tempo_inicio,
            "memoria_pico_mb": memoria_pico / (1024 * 1024),
            "orcamento_usado": orcamento_utilizado,
            "orcamento_restante": orcamento_restante,
            "ganho_marginal_total": ganho_marginal_total,
            "total_skus_beneficiados": len(skus_beneficiados),
            "itens_beneficiados": skus_beneficiados
        }

        return metricas