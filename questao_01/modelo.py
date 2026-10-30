"""Modelo de precificação da Questão 1 (Operação Caixa 90).

Tudo é orientado a dados: níveis de preço, piso de margem, meta e orçamento vêm do
arquivo de configuração da instância (ou de parâmetros de cenário), nunca de constantes.

Para cada SKU i e nível de preço l (multiplicador sobre ``preco_atual``):
    preco[i,l]   = preco_atual[i] * nivel[l]
    demanda      = max(0, demanda_base_90d * (1 - elasticidade * (nivel[l] - 1)))
    vendas[i,l]  = min(demanda, estoque)                       (trava de estoque)
    caixa_un     = preco*(1 - taxa_marketplace) - custo_unitario - custo_logistico
    margem       = caixa_un / preco                            (precisa ser >= piso)
    caixa[i,l]   = vendas * caixa_un
As tabelas (N x L) são pré-computadas uma única vez: avaliar uma política vira indexação.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def aplicar_cenario(df: pd.DataFrame, config: dict, *, meta_mult: float = 1.0,
                    orcamento_mult: float = 1.0, custo_log_mult: float = 1.0,
                    divida_mult: float = 1.0, remover_skus=()) -> tuple[pd.DataFrame, dict]:
    """Aplica um cenário-surpresa por parâmetros, sem tocar nos arquivos oficiais."""
    df = df.copy()
    config = dict(config)
    if remover_skus:
        df = df[~df["sku"].isin(list(remover_skus))]
    df["custo_logistico"] = df["custo_logistico"] * custo_log_mult
    config["meta_caixa_90d"] = config["meta_caixa_90d"] * meta_mult
    config["divida_total_90d"] = config["divida_total_90d"] * divida_mult
    config["orcamento_promocional"] = config["orcamento_promocional"] * orcamento_mult
    return df.reset_index(drop=True), config


@dataclass
class ModeloQ1:
    skus: list
    df: pd.DataFrame
    niveis: np.ndarray
    piso: float
    config: dict
    preco: np.ndarray       # (N, L)
    vendas: np.ndarray      # (N, L)
    caixa_un: np.ndarray    # (N, L)
    margem: np.ndarray      # (N, L)
    caixa: np.ndarray       # (N, L)
    viavel: np.ndarray      # (N, L) bool  (margem >= piso)

    @property
    def n(self) -> int:
        return len(self.skus)

    @property
    def n_niveis(self) -> int:
        return len(self.niveis)

    @property
    def ativos(self) -> np.ndarray:
        """Índices dos SKUs com pelo menos um preço permitido que respeite o piso de margem."""
        return np.flatnonzero(self.viavel.any(axis=1))

    @classmethod
    def construir(cls, df: pd.DataFrame, config: dict) -> "ModeloQ1":
        df = df.reset_index(drop=True)
        niveis = np.asarray(config["niveis_preco_relativos"], dtype=float)
        piso = float(config["margem_minima_percentual"])
        pref = df["preco_atual"].to_numpy(float)[:, None]
        preco = pref * niveis[None, :]
        demanda = np.maximum(
            0.0,
            df["demanda_base_90d"].to_numpy(float)[:, None]
            * (1.0 - df["elasticidade"].to_numpy(float)[:, None] * (niveis[None, :] - 1.0)),
        )
        vendas = np.minimum(demanda, df["estoque"].to_numpy(float)[:, None])
        caixa_un = (preco * (1.0 - df["taxa_marketplace"].to_numpy(float)[:, None])
                    - df["custo_unitario"].to_numpy(float)[:, None]
                    - df["custo_logistico"].to_numpy(float)[:, None])
        margem = caixa_un / preco
        return cls(skus=df["sku"].tolist(), df=df, niveis=niveis, piso=piso, config=config,
                   preco=preco, vendas=vendas, caixa_un=caixa_un, margem=margem,
                   caixa=vendas * caixa_un, viavel=margem >= piso)

    # ---- avaliação de políticas -------------------------------------------------
    def caixa_politica(self, idx: np.ndarray, posicoes: np.ndarray | None = None) -> float:
        """Caixa total de uma política (índice de nível por SKU). ``-inf`` se violar o piso."""
        pos = np.arange(self.n) if posicoes is None else np.asarray(posicoes)
        if not self.viavel[pos, idx].all():
            return -np.inf
        return float(self.caixa[pos, idx].sum())

    def resumo_politica(self, idx: np.ndarray) -> dict:
        """Métricas financeiras da política completa (SKUs inativos não vendem)."""
        ar = np.arange(self.n)
        ativo = self.viavel[ar, idx]
        vendas = np.where(ativo, self.vendas[ar, idx], 0.0)
        receita = vendas * self.preco[ar, idx]
        caixa = np.where(ativo, self.caixa[ar, idx], 0.0)
        estoque = self.df["estoque"].to_numpy(float)
        return {
            "caixa": float(caixa.sum()),
            "receita_bruta": float(receita.sum()),
            "margem_global": float(caixa.sum() / receita.sum()) if receita.sum() > 0 else 0.0,
            "unidades_vendidas": float(vendas.sum()),
            "estoque_inicial": float(estoque.sum()),
            "estoque_remanescente": float((estoque - vendas).sum()),
            "estoque_remanescente_pct": float((estoque - vendas).sum() / estoque.sum()),
            "valor_estoque_remanescente_custo": float(
                ((estoque - vendas) * self.df["custo_unitario"].to_numpy(float)).sum()),
        }


def status_meta(caixa: float, meta: float) -> dict:
    dif = caixa - meta
    return {"caixa": caixa, "meta": meta, "folga": dif, "meta_atingida": bool(dif >= 0),
            "status": "META ATINGIDA (folga)" if dif >= 0 else "META NÃO ATINGIDA (insuficiência)"}
