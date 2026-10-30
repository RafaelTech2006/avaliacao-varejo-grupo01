"""Etapa C — Greedy: distribui o orçamento de incentivos sobre a política de preços do GA.

Incentivo (cupom/cashback) em degraus cumulativos de ``niveis_incentivo`` (% do preço):
  * demanda sobe como se o preço caísse esse percentual (mesma elasticidade do modelo);
  * o gasto = preço * pct * (unidades incrementais + taxa_resgate_base * unidades que já seriam
    vendidas) — parte dos compradores "de qualquer jeito" também resgata o benefício;
  * o gasto sai do orçamento E é descontado do caixa (ganho líquido);
  * a margem unitária após o incentivo precisa respeitar o piso.
Regra gulosa: sempre dar o próximo degrau com maior (ganho líquido marginal / R$ investido),
somente se o ganho for positivo e couber no orçamento restante.
Complexidade: O((N*T) log N) tempo (heap), O(N*T) memória.
"""
from __future__ import annotations

import heapq
import time
import tracemalloc

import numpy as np

from questao_01.modelo import ModeloQ1


class GreedyIncentivos:
    def __init__(self, modelo: ModeloQ1, orcamento: float,
                 niveis_incentivo=(0.02, 0.04, 0.06), taxa_resgate_base: float = 0.25):
        self.m = modelo
        self.orcamento = float(orcamento)
        self.niveis = np.asarray(niveis_incentivo, dtype=float)
        self.rho = float(taxa_resgate_base)

    def _tabelas(self, idx: np.ndarray):
        """Para cada degrau t e SKU i: vendas, gasto, caixa líquido e viabilidade (T x N)."""
        m, df = self.m, self.m.df
        ar = np.arange(m.n)
        p = m.preco[ar, idx]
        vendas0 = m.vendas[ar, idx]
        cun = m.caixa_un[ar, idx]
        pref = df["preco_atual"].to_numpy(float)
        base = df["demanda_base_90d"].to_numpy(float)
        el = df["elasticidade"].to_numpy(float)
        est = df["estoque"].to_numpy(float)
        a = self.niveis[:, None]
        demanda = np.maximum(0.0, base[None, :] * (1.0 - el[None, :] * (p[None, :] * (1 - a) / pref[None, :] - 1.0)))
        vendas = np.minimum(demanda, est[None, :])
        vendas = np.maximum(vendas, vendas0[None, :])        # incentivo nunca reduz vendas
        gasto = p[None, :] * a * ((vendas - vendas0[None, :]) + self.rho * vendas0[None, :])
        liquido = vendas * cun[None, :] - gasto
        with np.errstate(divide="ignore", invalid="ignore"):
            margem = np.where(vendas > 0, (vendas * cun[None, :] - gasto) / (vendas * p[None, :]), 0.0)
        viavel = (margem >= m.piso) & m.viavel[ar, idx][None, :]
        return vendas, gasto, liquido, viavel, vendas0, vendas0 * cun

    def alocar(self, idx: np.ndarray) -> dict:
        tracemalloc.start()
        t0 = time.perf_counter()
        vendas, gasto, liquido, viavel, vendas0, caixa0 = self._tabelas(idx)
        T, N = vendas.shape
        atual = np.full(N, -1)                      # degrau já concedido (-1 = nenhum)
        restante = self.orcamento
        heap: list = []

        def melhor_passo(i):
            """Melhor próximo degrau de i (maior razão ganho/gasto marginal), ou None."""
            g_cur = caixa0[i] if atual[i] < 0 else liquido[atual[i], i]
            s_cur = 0.0 if atual[i] < 0 else gasto[atual[i], i]
            melhor = None
            for t in range(atual[i] + 1, T):
                dg, ds = liquido[t, i] - g_cur, gasto[t, i] - s_cur
                if viavel[t, i] and dg > 1e-9 and ds > 1e-9:
                    r = dg / ds
                    if melhor is None or r > melhor[0]:
                        melhor = (r, t, dg, ds)
            return melhor

        for i in range(N):
            p = melhor_passo(i)
            if p:
                heapq.heappush(heap, (-p[0], i, p[1], p[2], p[3]))
        while heap:
            negr, i, t, dg, ds = heapq.heappop(heap)
            if ds > restante + 1e-9:
                continue                             # não cabe; outros SKUs menores ainda podem caber
            atual[i], restante = t, restante - ds
            p = melhor_passo(i)
            if p:
                heapq.heappush(heap, (-p[0], i, p[1], p[2], p[3]))

        tempo = time.perf_counter() - t0
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        ar = np.arange(N)
        ativo = atual >= 0
        vendas_f = np.where(ativo, vendas[np.maximum(atual, 0), ar], vendas0)
        gasto_f = np.where(ativo, gasto[np.maximum(atual, 0), ar], 0.0)
        caixa_f = np.where(ativo, liquido[np.maximum(atual, 0), ar], caixa0)
        return {
            "degrau": atual, "vendas_final": vendas_f, "gasto": gasto_f, "caixa_final_sku": caixa_f,
            "caixa_antes_sku": caixa0, "vendas_antes_sku": vendas0,
            "orcamento": self.orcamento, "orcamento_usado": float(gasto_f.sum()),
            "orcamento_restante": float(restante), "ganho_liquido_total": float((caixa_f - caixa0).sum()),
            "skus_beneficiados": int(ativo.sum()), "tempo_s": tempo, "memoria_pico_mb": pico / 2**20,
        }
