import numpy as np
import pytest

from questao_02 import dynamic_programming as dp, greedy, monte_carlo as mc
from questao_02.modelo import Instancia, gerar_choques, demanda_semana


@pytest.fixture(scope="module")
def inst():
    return Instancia.de_arquivos("G01")


@pytest.fixture(scope="module")
def est(inst):
    return mc.estimar(inst, 400, seed=5)


def test_instancia_usa_dados_do_grupo(inst):
    assert inst.n == 24 and inst.n_semanas == 12
    assert inst.dividas == {4: 54546.73, 8: 70456.2, 12: 102275.12}
    assert inst.caixa_inicial == pytest.approx(24168.04)
    assert inst.niveis.tolist() == [0.82, 0.88, 0.94, 1.0, 1.06, 1.12]
    assert inst.orcamento_semanal == pytest.approx(6928.92)


def test_choques_reprodutiveis_e_media_um(inst):
    a = gerar_choques(np.random.default_rng(1), 4000, inst)
    b = gerar_choques(np.random.default_rng(1), 4000, inst)
    assert np.array_equal(a.ruido, b.ruido)
    assert a.ruido.mean() == pytest.approx(1.0, abs=0.01)
    assert a.fator_log.mean() == pytest.approx(1.0, abs=0.01)


def test_demanda_cai_quando_concorrente_barateia(inst):
    ruido = np.ones(inst.n)
    d_base = demanda_semana(inst, 1.0, ruido, np.ones(inst.n))
    d_conc = demanda_semana(inst, 1.0, ruido, np.full(inst.n, 0.92))
    assert (d_conc <= d_base + 1e-12).all() and (d_conc < d_base).any()


def test_pmf_soma_um(est):
    assert np.allclose(est.pmf.sum(-1), 1.0)


def test_estimacao_reprodutivel(inst):
    a, b = mc.estimar(inst, 200, seed=9), mc.estimar(inst, 200, seed=9)
    assert np.array_equal(a.pmf, b.pmf)
    c = mc.estimar(inst, 200, seed=10)
    assert not np.array_equal(a.pmf, c.pmf)


def test_matriz_vendas_linhas_sao_distribuicoes(est):
    K = dp.matriz_vendas(est.pmf[0, 0, 0], 31)
    assert np.allclose(K.sum(-1), 1.0)
    assert (K[:, 0, 1:] == 0).all()           # estoque 0 -> vendas 0


def test_dp_trajetoria_completa_e_estoque_nunca_negativo(inst, est):
    sol = dp.otimizar(inst, est, lam=0.5, beta=0.1)
    assert sol["nivel_idx"].shape == (inst.n, 12)
    assert ((sol["nivel_idx"] >= 0) & (sol["nivel_idx"] < inst.n_niveis)).all()
    assert (sol["estoque_esperado"] >= 0).all()
    assert (np.diff(sol["estoque_esperado"], axis=1) <= 0).all()      # estoque só diminui


def test_dp_sem_estoque_valor_zero(inst, est):
    pequena = inst.com_cenario(remover_skus=inst.skus[1:])
    pequena.estoque0[:] = 0
    e = mc.estimar(pequena, 100, seed=1)
    r = dp.resolver_sku(0, pequena, e, 0.0, np.ones(12))
    assert r["valor"] == 0.0


def test_dp_um_nivel_domina(inst):
    """Com um único SKU de demanda inelástica, o melhor preço é o mais alto."""
    um = inst.com_cenario(remover_skus=inst.skus[1:])
    um.elast[:] = 0.0
    e = mc.estimar(um, 300, seed=2)
    r = dp.resolver_sku(0, um, e, 0.0, np.ones(12))
    com_estoque = [n for n, st in zip(r["niveis"], r["estoque_esperado"]) if st > 0]   # sem estoque todo preço empata
    assert com_estoque and set(com_estoque) == {um.n_niveis - 1}


def test_greedy_respeita_orcamento_semanal(inst, est):
    sol = dp.otimizar(inst, est)
    for orc in (0.0, 500.0, inst.orcamento_semanal):
        plano, _ = greedy.alocar(inst, est, sol["nivel_idx"], orcamento_semanal=orc)
        gasto_sem = (plano.custo_destaque * plano.destaque).sum(0)
        assert (gasto_sem <= orc + 1e-6).all()
    plano0, _ = greedy.alocar(inst, est, sol["nivel_idx"], orcamento_semanal=0.0)
    assert not plano0.destaque.any()


def test_validacao_reprodutivel_e_seed_independente(inst, est):
    sol = dp.otimizar(inst, est)
    plano, _ = greedy.alocar(inst, est, sol["nivel_idx"])
    v1 = mc.validar(inst, plano, 600, seed=77)
    v2 = mc.validar(inst, plano, 600, seed=77)
    v3 = mc.validar(inst, plano, 600, seed=78)
    assert np.array_equal(v1["caixa_final_cenarios"], v2["caixa_final_cenarios"])
    assert not np.array_equal(v1["caixa_final_cenarios"], v3["caixa_final_cenarios"])
    assert all(0 <= p <= 1 for p in v1["prob_pagamento"].values())


def test_validacao_respeita_estoque(inst, est):
    sol = dp.otimizar(inst, est)
    plano, _ = greedy.alocar(inst, est, sol["nivel_idx"])
    v = mc.validar(inst, plano, 300, seed=3)
    assert v["estoque_final_medio_un"] >= 0
    assert v["estoque_final_medio_un"] <= inst.estoque0.sum()
    assert (v["estoque_medio_por_semana"] >= -1e-9).all()


def test_dividas_maiores_reduzem_probabilidade(inst, est):
    sol = dp.otimizar(inst, est)
    plano, _ = greedy.alocar(inst, est, sol["nivel_idx"])
    normal = mc.validar(inst, plano, 800, seed=11)
    apertada = mc.validar(inst.com_cenario(divida_mult=3.0), plano, 800, seed=11)
    assert apertada["prob_todas"] < normal["prob_todas"]
    # dívida impossível -> nenhuma chance
    imposs = mc.validar(inst.com_cenario(divida_mult=100.0), plano, 300, seed=11)
    assert imposs["prob_todas"] == 0.0


def test_cenario_remove_sku_e_replica(inst):
    assert inst.com_cenario(remover_skus=[inst.skus[0]]).n == inst.n - 1
    assert inst.replicar(4).n == 4 * inst.n


def test_dp_e_melhor_que_politica_fixa_em_cenario_estressado(inst, est):
    sol = dp.otimizar(inst, est)
    plano, _ = greedy.alocar(inst, est, sol["nivel_idx"])
    fixa = mc.PlanoCompleto(np.full_like(sol["nivel_idx"], 3), np.zeros_like(plano.destaque), plano.custo_destaque)
    estresse = inst.com_cenario(divida_mult=1.3, concorrente_delta=-0.08, custo_log_mult=1.15)
    pd_ = mc.validar(estresse, plano, 1000, seed=21)["prob_todas"]
    pf = mc.validar(estresse, fixa, 1000, seed=21)["prob_todas"]
    assert pd_ > pf
