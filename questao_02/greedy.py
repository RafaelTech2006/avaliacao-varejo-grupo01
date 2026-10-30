"""Etapa C — Greedy semanal: aloca destaques/mídia/frete promocional (recurso escasso) a cada semana.

Para cada semana w (em ordem), usando o plano de preços da DP e o estoque esperado do SKU:
    ganho(i,w)  = ( E[min(D*(1+u), s)] - E[min(D, s)] ) * margem_média  -  custo_destaque(i,w)
    razão(i,w)  = ganho / custo_destaque     (benefício líquido por real investido)
Ordena por razão decrescente e concede destaques enquanto couber no orçamento semanal e ganho > 0.
O estoque esperado é atualizado semana a semana (um destaque faz o estoque girar mais rápido).
Tempo O(W * N log N); memória O(N).
"""
from __future__ import annotations

import numpy as np

from questao_02.modelo import Instancia, custo_promo
from questao_02.monte_carlo import Estatisticas, PlanoCompleto


def vendas_esperadas(pmf: np.ndarray, s: int) -> float:
    """E[min(D, s)] a partir da pmf de demanda inteira."""
    s = int(min(s, len(pmf) - 1))
    return float((np.arange(s) * pmf[:s]).sum() + s * pmf[s:].sum())


def alocar(inst: Instancia, est: Estatisticas, nivel_idx: np.ndarray, estoque_ini: np.ndarray | None = None,
           custo_pct: float = 0.05, orcamento_semanal: float | None = None) -> tuple[PlanoCompleto, list]:
    W, N = inst.n_semanas, inst.n
    orc = inst.orcamento_semanal if orcamento_semanal is None else orcamento_semanal
    estoque = inst.estoque0.astype(float).copy() if estoque_ini is None else estoque_ini.astype(float).copy()
    destaque = np.zeros((N, W), dtype=bool)
    custos = np.zeros((N, W))
    registro = []
    for w in range(W):
        m = nivel_idx[:, w]
        custo = custo_promo(inst, m, custo_pct)
        ganho = np.zeros(N)
        e_base = np.zeros(N)
        e_prom = np.zeros(N)
        for i in range(N):
            e_base[i] = vendas_esperadas(est.pmf[0, w, i, m[i]], estoque[i])
            e_prom[i] = vendas_esperadas(est.pmf[1, w, i, m[i]], estoque[i])
            ganho[i] = (e_prom[i] - e_base[i]) * est.media_margem[w, i, m[i]] - custo[i]
        razao = np.where(custo > 0, ganho / custo, -np.inf)
        restante = orc
        for i in np.argsort(-razao):                       # ordenação gulosa por benefício/real
            if ganho[i] <= 0:
                break
            if custo[i] <= restante:
                restante -= custo[i]
                destaque[i, w] = True
                registro.append({"semana": w + 1, "sku": inst.skus[i], "custo": float(custo[i]),
                                 "ganho_liquido_esperado": float(ganho[i]), "razao": float(razao[i])})
        custos[:, w] = custo
        estoque = np.maximum(0.0, estoque - np.rint(np.where(destaque[:, w], e_prom, e_base)))
    return PlanoCompleto(nivel_idx=nivel_idx, destaque=destaque, custo_destaque=custos), registro
