"""Etapa A (e validação final) — Monte Carlo.

* ``estimar``: usa ``n`` cenários para estimar, por (semana, SKU, nível de preço), a distribuição das
  vendas semanais (pmf de unidades) com e sem destaque, e os momentos da margem unitária. Alimenta a DP e o Greedy.
* ``validar``: simula o PLANO COMPLETO (preços da DP + destaques do Greedy) em cenários INDEPENDENTES
  (seed distinta) com controle de estoque e de caixa acumulado, e mede a probabilidade de pagar cada dívida.
Ambos processam os cenários em blocos para limitar memória: O(bloco*W*N) em vez de O(C*W*N).
Tempo: estimar O(C*W*N*L); validar O(C*W*N). Memória: O(bloco*W*N + W*N*L*S).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from questao_02.modelo import Instancia, demanda_semana, gerar_choques


@dataclass
class Estatisticas:
    pmf: np.ndarray        # (2, W, N, L, S+1): [0]=sem destaque, [1]=com destaque; P(vendas desejadas = j)
    media_margem: np.ndarray   # (W, N, L)
    margem2: np.ndarray        # (W, N, L)  E[margem^2]
    estoque_max: int
    n_cenarios: int
    uplift: float


def estimar(inst: Instancia, n_cenarios: int = 2000, seed: int = 0, uplift: float = 0.20,
            bloco: int = 500) -> Estatisticas:
    rng = np.random.default_rng(seed)
    W, N, L = inst.n_semanas, inst.n, inst.n_niveis
    S = int(inst.estoque0.max())
    cont = np.zeros((2, L, W * N * (S + 1)))
    soma_f = np.zeros((W, N))
    soma_f2 = np.zeros((W, N))
    base_idx = (np.arange(W)[None, :, None] * N + np.arange(N)[None, None, :]) * (S + 1)
    feitos = 0
    while feitos < n_cenarios:
        c = min(bloco, n_cenarios - feitos)
        ch = gerar_choques(rng, c, inst)
        soma_f += ch.fator_log.sum(0)
        soma_f2 += (ch.fator_log ** 2).sum(0)
        for l, m in enumerate(inst.niveis):
            d = demanda_semana(inst, m, ch.ruido, ch.fator_pc)
            for j, mult in enumerate((1.0, 1.0 + uplift)):
                k = np.clip(np.rint(d * mult), 0, S).astype(np.int64)
                cont[j, l] += np.bincount((base_idx + k).ravel(), minlength=W * N * (S + 1))
        feitos += c
    pmf = (cont / n_cenarios).reshape(2, L, W, N, S + 1).transpose(0, 2, 3, 1, 4)
    ef, ef2 = soma_f / n_cenarios, soma_f2 / n_cenarios                      # (W, N)
    a = (inst.pref[:, None] * inst.niveis[None, :] * (1 - inst.taxa[:, None]) - inst.custo_unit[:, None])  # (N, L)
    cl = inst.custo_log[None, :, None]
    media = a[None] - cl * ef[:, :, None]                                    # E[a - cl*f]
    m2 = a[None] ** 2 - 2 * a[None] * cl * ef[:, :, None] + cl ** 2 * ef2[:, :, None]
    return Estatisticas(pmf=pmf, media_margem=media, margem2=m2, estoque_max=S,
                        n_cenarios=n_cenarios, uplift=uplift)


@dataclass
class PlanoCompleto:
    nivel_idx: np.ndarray      # (N, W) índice do nível de preço por semana
    destaque: np.ndarray       # (N, W) bool
    custo_destaque: np.ndarray  # (N, W) R$


def validar(inst: Instancia, plano: PlanoCompleto, n_cenarios: int = 5000, seed: int = 10_000,
            uplift: float = 0.20, bloco: int = 1000) -> dict:
    """Simula o plano completo em cenários novos. Dívidas são cumulativas: o compromisso da semana d é
    pago no prazo se caixa_inicial + caixa_gerado(1..d) - dívidas(<=d) >= 0."""
    rng = np.random.default_rng(seed)
    W, N = inst.n_semanas, inst.n
    mult = inst.niveis[plano.nivel_idx]                                    # (N, W)
    preco = inst.pref[:, None] * mult
    prom_custo_sem = (plano.custo_destaque * plano.destaque).sum(0)       # (W,)
    saldo = np.zeros((n_cenarios, W))        # caixa acumulado (antes de pagar dívidas)
    receita = np.zeros(n_cenarios)
    contrib = np.zeros(n_cenarios)
    estoque_final = np.zeros((n_cenarios, N))
    estoque_medio_sem = np.zeros((W + 1, N))
    feitos = 0
    while feitos < n_cenarios:
        c = min(bloco, n_cenarios - feitos)
        ch = gerar_choques(rng, c, inst)
        est = np.tile(inst.estoque0.astype(float), (c, 1))
        acum = np.full(c, inst.caixa_inicial)
        estoque_medio_sem[0] += est.sum(0)
        for w in range(W):
            up = np.where(plano.destaque[:, w], 1.0 + uplift, 1.0)
            d = demanda_semana(inst, mult[:, w], ch.ruido[:, w, :], ch.fator_pc[:, w, :]) * up
            k = np.minimum(np.rint(d), est)
            margem = preco[:, w] * (1 - inst.taxa) - inst.custo_unit - inst.custo_log * ch.fator_log[:, w, :]
            fluxo = (k * margem).sum(1)
            acum = acum + fluxo - prom_custo_sem[w]
            saldo[feitos:feitos + c, w] = acum
            receita[feitos:feitos + c] += (k * preco[:, w]).sum(1)
            contrib[feitos:feitos + c] += fluxo
            est = est - k
            estoque_medio_sem[w + 1] += est.sum(0)
        estoque_final[feitos:feitos + c] = est
        feitos += c
    estoque_medio_sem /= n_cenarios

    div_acum = np.zeros(W)
    for d, v in inst.dividas.items():
        div_acum[d - 1:] += v
    folga = saldo - div_acum[None, :]                                     # saldo após dívidas vencidas até w
    venc = inst.semanas_vencimento
    ok = {d: folga[:, d - 1] >= 0 for d in venc}
    todas = np.all([ok[d] for d in venc], axis=0)
    caixa_final = folga[:, -1]
    q = lambda p: float(np.percentile(caixa_final, p))
    return {
        "n_cenarios": n_cenarios, "seed": seed,
        "prob_pagamento": {int(d): float(ok[d].mean()) for d in venc},
        "prob_todas": float(todas.mean()), "cenarios_falha_qualquer": int((~todas).sum()),
        "cenarios_falha_por_semana": {int(d): int((~ok[d]).sum()) for d in venc},
        "caixa_final_medio": float(caixa_final.mean()), "caixa_final_p5": q(5), "caixa_final_p10": q(10),
        "caixa_final_p50": q(50), "caixa_final_p95": q(95),
        "caixa_gerado_medio": float(contrib.mean() - prom_custo_sem.sum()),
        "custo_destaques": float(prom_custo_sem.sum()),
        "margem_media": float((contrib.sum() - prom_custo_sem.sum() * n_cenarios) / receita.sum()),
        "receita_media": float(receita.mean()),
        "estoque_final_medio_un": float(estoque_final.sum(1).mean()),
        "estoque_final_pct": float(estoque_final.sum(1).mean() / inst.estoque0.sum()),
        "estoque_medio_por_semana": estoque_medio_sem,                    # (W+1, N)
        "saldo_medio_por_semana": folga.mean(0), "saldo_p10_por_semana": np.percentile(folga, 10, axis=0),
        "caixa_final_cenarios": caixa_final,
    }
