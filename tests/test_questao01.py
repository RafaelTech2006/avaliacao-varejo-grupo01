import numpy as np
import pandas as pd
import pytest

from comum.dados import carregar_instancia_bruta
from questao_01.brute_force import brute_force, selecionar_subconjunto
from questao_01.genetic_algorithm import (AlgoritmoGenetico, ConfigGA, gap_percentual, otimo_separavel)
from questao_01.greedy import GreedyIncentivos
from questao_01.modelo import ModeloQ1, aplicar_cenario, status_meta
from questao_01 import otimizacao


@pytest.fixture(scope="module")
def inst():
    return carregar_instancia_bruta(1, "G01")


@pytest.fixture()
def mini():
    """3 SKUs, 3 níveis — pequeno o bastante para conferir à mão."""
    df = pd.DataFrame({
        "sku": ["A", "B", "C"], "categoria": "x", "custo_unitario": [50.0, 80.0, 90.0],
        "preco_atual": [100.0, 120.0, 100.0], "estoque": [1000, 40, 100], "demanda_base_90d": [100, 100, 100],
        "elasticidade": [1.0, 1.0, 1.0], "taxa_marketplace": [0.1, 0.1, 0.1], "custo_logistico": [5.0, 5.0, 5.0],
        "idade_estoque_dias": 10})
    cfg = {"niveis_preco_relativos": [0.9, 1.0, 1.1], "margem_minima_percentual": 0.10,
           "meta_caixa_90d": 1000.0, "divida_total_90d": 900.0, "orcamento_promocional": 500.0}
    return df, cfg


def test_modelo_calculo_manual(mini):
    m = ModeloQ1.construir(*mini)
    # SKU A, preço 100 (nível 1.0): demanda 100, vendas 100, caixa_un = 90 - 50 - 5 = 35
    assert m.caixa[0, 1] == pytest.approx(100 * 35)
    # trava de estoque: B no preço 120*0.9=108 -> demanda 110, estoque 40
    assert m.vendas[1, 0] == 40
    # piso de margem: C a 90 -> caixa_un = 81-90-5 < 0 => inviável
    assert not m.viavel[2, 0]


def test_brute_force_acha_otimo_manual(mini):
    m = ModeloQ1.construir(*mini)
    r = brute_force(m, np.arange(3))
    esperado = sum(np.where(m.viavel[i], m.caixa[i], -np.inf).max() for i in range(3))
    assert r["caixa"] == pytest.approx(esperado)
    assert r["combinacoes"] == int(np.prod(m.viavel.sum(1)))


def test_ga_reproduzivel_por_seed_e_valida_no_subconjunto(inst):
    m = ModeloQ1.construir(*inst)
    sub = selecionar_subconjunto(m, 6, seed=3)
    bf = brute_force(m, sub)
    r1 = AlgoritmoGenetico(m, ConfigGA(), posicoes=sub, seed=7).otimizar()
    r2 = AlgoritmoGenetico(m, ConfigGA(), posicoes=sub, seed=7).otimizar()
    assert r1.fitness == r2.fitness and np.array_equal(r1.idx, r2.idx)
    assert gap_percentual(bf["caixa"], r1.fitness) == pytest.approx(0.0, abs=1e-6)


def test_ga_catalogo_completo_proximo_do_otimo_exato(inst):
    m = ModeloQ1.construir(*inst)
    at = m.ativos
    r = AlgoritmoGenetico(m, ConfigGA(tamanho_populacao=100, geracoes=200), posicoes=at, seed=1).otimizar()
    exato = m.caixa[at, otimo_separavel(m)[at]].sum()
    assert r.fitness <= exato + 1e-6
    assert gap_percentual(exato, r.fitness) < 0.5


def test_ga_historico_com_elitismo_nao_decresce(inst):
    m = ModeloQ1.construir(*inst)
    r = AlgoritmoGenetico(m, ConfigGA(n_elite=2, paciencia=10**6), posicoes=m.ativos, seed=2).otimizar()
    melhores = [h["melhor_geracao"] for h in r.historico]
    assert all(b >= a - 1e-9 for a, b in zip(melhores, melhores[1:]))


def test_ga_nao_escolhe_preco_que_viola_margem(inst):
    m = ModeloQ1.construir(*inst)
    r = AlgoritmoGenetico(m, ConfigGA(), posicoes=m.ativos, seed=4).otimizar()
    assert m.viavel[m.ativos, r.idx].all()


def test_subconjunto_invalido(inst):
    m = ModeloQ1.construir(*inst)
    with pytest.raises(ValueError):
        selecionar_subconjunto(m, 0)
    with pytest.raises(ValueError):
        selecionar_subconjunto(m, 10_000)


def test_greedy_respeita_orcamento_e_nao_gera_ganho_negativo(inst):
    m = ModeloQ1.construir(*inst)
    idx = np.zeros(m.n, dtype=int)
    idx[m.ativos] = AlgoritmoGenetico(m, ConfigGA(), posicoes=m.ativos, seed=1).otimizar().idx
    for orc in (0.0, 1_000.0, 20_000.0, 1e9):
        g = GreedyIncentivos(m, orc).alocar(idx)
        assert g["orcamento_usado"] <= orc + 1e-6
        assert g["ganho_liquido_total"] >= -1e-6
    assert GreedyIncentivos(m, 0.0).alocar(idx)["skus_beneficiados"] == 0


def test_greedy_mais_orcamento_nunca_piora(inst):
    m = ModeloQ1.construir(*inst)
    idx = otimo_separavel(m)
    ganhos = [GreedyIncentivos(m, o).alocar(idx)["ganho_liquido_total"] for o in (1e3, 5e3, 2e4, 1e6)]
    assert all(b >= a - 1e-6 for a, b in zip(ganhos, ganhos[1:]))


def test_cenarios_por_parametro(inst):
    df, cfg = inst
    df2, cfg2 = aplicar_cenario(df, cfg, meta_mult=1.10, orcamento_mult=0.75, custo_log_mult=1.15,
                                remover_skus=["Q1-001"])
    assert cfg2["meta_caixa_90d"] == pytest.approx(cfg["meta_caixa_90d"] * 1.10)
    assert cfg2["orcamento_promocional"] == pytest.approx(cfg["orcamento_promocional"] * 0.75)
    assert len(df2) == len(df) - 1 and "Q1-001" not in set(df2["sku"])
    assert df2["custo_logistico"].iloc[0] == pytest.approx(df["custo_logistico"].iloc[1] * 1.15)


def test_status_meta():
    assert status_meta(110, 100)["meta_atingida"] and status_meta(110, 100)["folga"] == 10
    assert not status_meta(90, 100)["meta_atingida"] and status_meta(90, 100)["folga"] == -10


def test_otimizacao_mantem_resultado(inst):
    m = ModeloQ1.construir(*inst)
    r = otimizacao.comparar(m, n_avaliacoes=20)
    assert r["resultados_iguais"] and r["speedup"] > 1
