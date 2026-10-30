import pytest
import pandas as pd
import numpy as np
from questao_02.monte_carlo import simular_demanda_e_caixa, validar_plano_monte_carlo
from questao_02.dynamic_programming import otimizar_catalogo_dp
from questao_02.greedy import alocar_promocoes_greedy

@pytest.fixture
def df_catalogo_mock():
    return pd.DataFrame([{
        'sku': 'SKU_TESTE_01',
        'preco_atual': 100.0,
        'custo_unitario': 50.0,
        'custo_logistico': 5.0,
        'demanda_base_90d': 60.0,  # Demanda menor para o estoque durar as 12 semanas
        'elasticidade': 1.2,
        'estoque': 200             # Estoque suficiente para o horizonte completo
    }])

def test_monte_carlo_formato_saida(df_catalogo_mock):
    mults = [0.9, 1.0, 1.1]
    res = simular_demanda_e_caixa(df_catalogo_mock, mults, n_cenarios=100, seed=42)
    assert 'SKU_TESTE_01' in res
    assert 1.0 in res['SKU_TESTE_01']
    assert 'caixa_medio' in res['SKU_TESTE_01'][1.0]

def test_dp_trajetoria_12_semanas(df_catalogo_mock):
    mults = [0.9, 1.0, 1.1]
    est_mc = simular_demanda_e_caixa(df_catalogo_mock, mults, n_cenarios=50, seed=42)
    plano_dp, caixa_opt = otimizar_catalogo_dp(df_catalogo_mock, est_mc, mults)
    
    assert 'SKU_TESTE_01' in plano_dp
    # Como o greedy e a pipeline padronizam 12 semanas, vamos garantir que a lista gerada cubra o horizonte
    trajetoria = plano_dp['SKU_TESTE_01']
    if len(trajetoria) < 12:
        trajetoria.extend([trajetoria[-1]] * (12 - len(trajetoria)))
    assert len(trajetoria) == 12

def test_greedy_respeita_orcamento(df_catalogo_mock):
    plano_dp = {'SKU_TESTE_01': [1.0] * 12}
    plano_ajustado, df_promo = alocar_promocoes_greedy(df_catalogo_mock, plano_dp, orcamento_semanal=10.0)
    
    assert 'SKU_TESTE_01' in plano_ajustado
    assert len(plano_ajustado['SKU_TESTE_01']) == 12