"""Etapa B — Programação Dinâmica (bottom-up) para a trajetória semanal de preços de cada SKU.

Formulação (por SKU):
  Estado      : (w, s)  semana w = 1..W e estoque remanescente s = 0..estoque_inicial  (exato, sem discretizar)
  Decisão     : m  nível de preço da semana w  (um dos L níveis relativos)
  Aleatório   : vendas k = min(D, s), com D ~ pmf estimada pelo Monte Carlo para (w, SKU, m)
  Transição   : s' = s - k
  Recompensa  : r(w,s,m) = peso_w * ( E[k*margem] - lambda * desvio_padrao[k*margem] )
  Objetivo    : V(w,s) = max_m  r(w,s,m) + sum_k P(k | w,s,m) * V(w+1, s-k),   V(W+1, .) = 0
  Reutilização: V(w+1,.) é calculado UMA vez e consultado por todos os (s, m, k) da semana w; sem memoização o
                número de trajetórias seria L^W.
  peso_w = 1 + beta * (nº de vencimentos em semanas >= w)  -> caixa gerado cedo vale mais (dívidas nas semanas 4/8/12)
Tempo O(N*W*L*S^2); memória O(N*W*S) para a política (O(L*S^2) temporário por semana).
"""
from __future__ import annotations

import numpy as np

from questao_02.modelo import Instancia
from questao_02.monte_carlo import Estatisticas


def matriz_vendas(pmf: np.ndarray, n_est: int) -> np.ndarray:
    """K[m, s, k] = P(vendas = k | estoque s, nível m) = pmf[m,k] se k<s ; P(D>=s) se k=s ; 0 se k>s."""
    L = pmf.shape[0]
    cauda = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]                      # P(D >= j)
    s = np.arange(n_est)
    K = np.where(s[None, None, :] < s[None, :, None], pmf[:, None, :n_est], 0.0)
    K[:, s, s] = cauda[:, :n_est]
    return K


def pesos_semanais(inst: Instancia, beta: float) -> np.ndarray:
    venc = inst.semanas_vencimento
    return np.array([1.0 + beta * sum(1 for d in venc if d >= w) for w in range(1, inst.n_semanas + 1)])


def resolver_sku(i: int, inst: Instancia, est: Estatisticas, lam: float, pesos: np.ndarray) -> dict:
    W = inst.n_semanas
    n = int(inst.estoque0[i]) + 1
    ks = np.arange(n, dtype=float)
    idx_prox = np.maximum(np.arange(n)[:, None] - np.arange(n)[None, :], 0)   # s - k
    V = np.zeros((W + 2, n))
    politica = np.zeros((W + 2, n), dtype=int)
    Ek_all = {}
    for w in range(W, 0, -1):                                                 # bottom-up
        K = matriz_vendas(est.pmf[0, w - 1, i], n)                            # (L, n, n)
        Ek, Ek2 = K @ ks, K @ ks ** 2
        media = Ek * est.media_margem[w - 1, i][:, None]
        var = Ek2 * est.margem2[w - 1, i][:, None] - media ** 2
        recompensa = pesos[w - 1] * (media - lam * np.sqrt(np.maximum(var, 0.0)))
        Q = recompensa + (K * V[w + 1][idx_prox][None]).sum(-1)               # (L, n)
        politica[w], V[w] = Q.argmax(0), Q.max(0)
        Ek_all[w] = Ek
    # reconstrução do caminho ótimo (trajetória esperada de preços e estoque)
    s = n - 1
    niveis, estoques = [], [s]
    for w in range(1, W + 1):
        m = int(politica[w, s])
        niveis.append(m)
        s = int(max(0, s - round(float(Ek_all[w][m, s]))))
        estoques.append(s)
    return {"valor": float(V[1, n - 1]), "niveis": niveis, "estoque_esperado": estoques,
            "subproblemas": W * n, "transicoes": W * inst.n_niveis * n * n}


def otimizar(inst: Instancia, est: Estatisticas, lam: float = 0.5, beta: float = 0.10) -> dict:
    pesos = pesos_semanais(inst, beta)
    res = [resolver_sku(i, inst, est, lam, pesos) for i in range(inst.n)]
    return {
        "nivel_idx": np.array([r["niveis"] for r in res]),                    # (N, W)
        "estoque_esperado": np.array([r["estoque_esperado"] for r in res]),   # (N, W+1)
        "valor": float(sum(r["valor"] for r in res)),
        "subproblemas": int(sum(r["subproblemas"] for r in res)),
        "transicoes": int(sum(r["transicoes"] for r in res)),
        "trajetorias_sem_memo_por_sku": inst.n_niveis ** inst.n_semanas,
    }
