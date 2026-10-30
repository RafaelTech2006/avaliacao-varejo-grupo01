"""Modelo da Questão 2 (Plano de Sobrevivência em 12 Semanas) — dados, choques e demanda.

Modelo semanal para o SKU i, semana w, nível de preço m (multiplicador sobre preco_atual):
    preço do concorrente  pc[w] = preco_concorrente_base * exp(passeio aleatório, sigma_preco_concorrente)
    demanda               D = base_semanal * max(0, 1 - elasticidade*(m/fator_pc[w] - 1)) * ruido[w]
                          ruido ~ lognormal(média 1, sigma_demanda);  m/fator_pc = preço relativo ao concorrente
    custo logístico       cl[w] = custo_logistico_base * fator[w]  (lognormal média 1, sigma_custo_logistico)
    vendas                k = min(arredondar(D * (1+uplift se destaque)), estoque)
    caixa unitário        preco*(1 - taxa) - custo_unitario - cl[w]      (mesma definição do gerador)
Choques têm um componente comum a todos os SKUs (correlação RHO) + um idiossincrático:
isso é o que gera risco de cauda real no caixa agregado.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from comum.dados import DATA_DIR, carregar_instancia_bruta

RHO = 0.5   # fração da variância dos choques que é comum a todos os SKUs


@dataclass
class Instancia:
    skus: list
    categorias: list
    pref: np.ndarray
    custo_unit: np.ndarray
    taxa: np.ndarray
    custo_log: np.ndarray
    estoque0: np.ndarray        # int
    base_sem: np.ndarray
    elast: np.ndarray
    pc0: np.ndarray
    niveis: np.ndarray
    n_semanas: int
    caixa_inicial: float
    dividas: dict               # {semana: valor}
    sigma_dem: float
    sigma_conc: float
    sigma_log: float
    orcamento_semanal: float
    limite_risco: float

    @property
    def n(self) -> int:
        return len(self.skus)

    @property
    def n_niveis(self) -> int:
        return len(self.niveis)

    @property
    def semanas_vencimento(self) -> list:
        return sorted(self.dividas)

    @classmethod
    def de_arquivos(cls, grupo: str = "G01", data_dir: Path = DATA_DIR) -> "Instancia":
        df, cfg = carregar_instancia_bruta(2, grupo, data_dir)
        var = cfg["variabilidade"]
        return cls(
            skus=df["sku"].tolist(), categorias=df["categoria"].tolist(),
            pref=df["preco_atual"].to_numpy(float), custo_unit=df["custo_unitario"].to_numpy(float),
            taxa=df["taxa_marketplace"].to_numpy(float), custo_log=df["custo_logistico_base"].to_numpy(float),
            estoque0=df["estoque_inicial"].to_numpy(int), base_sem=df["demanda_base_semanal"].to_numpy(float),
            elast=df["elasticidade"].to_numpy(float), pc0=df["preco_concorrente_base"].to_numpy(float),
            niveis=np.asarray(cfg["niveis_preco_relativos"], float), n_semanas=int(cfg["numero_semanas"]),
            caixa_inicial=float(cfg["caixa_inicial"]),
            dividas={int(k.split("_")[1]): float(v) for k, v in cfg["dividas"].items()},
            sigma_dem=float(var["sigma_demanda"]), sigma_conc=float(var["sigma_preco_concorrente"]),
            sigma_log=float(var["sigma_custo_logistico"]),
            orcamento_semanal=float(cfg["orcamento_promocional_semanal"]),
            limite_risco=float(cfg["limite_risco_falha"]))

    def com_cenario(self, *, divida_mult=1.0, concorrente_delta=0.0, custo_log_mult=1.0, orcamento_mult=1.0,
                    sigma_mult=1.0, remover_skus=()) -> "Instancia":
        """Cenário-surpresa por parâmetros (sem editar os arquivos oficiais)."""
        keep = np.array([s not in set(remover_skus) for s in self.skus])
        sel = lambda a: a[keep]
        return replace(
            self, skus=[s for s, k in zip(self.skus, keep) if k],
            categorias=[c for c, k in zip(self.categorias, keep) if k],
            pref=sel(self.pref), custo_unit=sel(self.custo_unit), taxa=sel(self.taxa),
            custo_log=sel(self.custo_log) * custo_log_mult, estoque0=sel(self.estoque0),
            base_sem=sel(self.base_sem), elast=sel(self.elast), pc0=sel(self.pc0) * (1 + concorrente_delta),
            dividas={k: v * divida_mult for k, v in self.dividas.items()},
            sigma_dem=self.sigma_dem * sigma_mult, sigma_conc=self.sigma_conc * sigma_mult,
            sigma_log=self.sigma_log * sigma_mult, orcamento_semanal=self.orcamento_semanal * orcamento_mult)

    def replicar(self, fator: int) -> "Instancia":
        """Replica o catálogo ``fator`` vezes (e escala as dívidas) — usado no teste de escalabilidade."""
        if fator == 1:
            return self
        rep = lambda a: np.tile(a, fator)
        return replace(
            self, skus=[f"{s}#{r}" for r in range(fator) for s in self.skus],
            categorias=self.categorias * fator, pref=rep(self.pref), custo_unit=rep(self.custo_unit),
            taxa=rep(self.taxa), custo_log=rep(self.custo_log), estoque0=rep(self.estoque0),
            base_sem=rep(self.base_sem), elast=rep(self.elast), pc0=rep(self.pc0),
            caixa_inicial=self.caixa_inicial * fator, dividas={k: v * fator for k, v in self.dividas.items()},
            orcamento_semanal=self.orcamento_semanal * fator)


@dataclass
class Choques:
    ruido: np.ndarray      # (C, W, N) multiplicador de demanda (média 1)
    fator_pc: np.ndarray   # (C, W, N) preço do concorrente relativo ao base (passeio aleatório)
    fator_log: np.ndarray  # (C, W, N) multiplicador do custo logístico (média 1)


def gerar_choques(rng: np.random.Generator, n_cen: int, inst: Instancia) -> Choques:
    W, N = inst.n_semanas, inst.n

    def eps():
        comum = rng.standard_normal((n_cen, W, 1))
        idio = rng.standard_normal((n_cen, W, N))
        return np.sqrt(RHO) * comum + np.sqrt(1 - RHO) * idio

    ruido = np.exp(inst.sigma_dem * eps() - inst.sigma_dem ** 2 / 2)
    fator_pc = np.exp(np.cumsum(inst.sigma_conc * eps(), axis=1))
    fator_log = np.exp(inst.sigma_log * eps() - inst.sigma_log ** 2 / 2)
    return Choques(ruido, fator_pc, fator_log)


def demanda_semana(inst: Instancia, mult, ruido, fator_pc):
    """Demanda contínua. ``mult`` pode ser escalar ou (N,); ruido/fator_pc: (..., N) broadcastável."""
    rel = mult / fator_pc
    return inst.base_sem * np.maximum(0.0, 1.0 - inst.elast * (rel - 1.0)) * ruido


def custo_promo(inst: Instancia, nivel_idx: np.ndarray, custo_pct: float) -> np.ndarray:
    """Custo de um destaque/frete promocional: fração da receita nominal semanal do SKU no preço do plano."""
    return custo_pct * inst.pref * inst.niveis[nivel_idx] * inst.base_sem
