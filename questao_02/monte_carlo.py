import numpy as np
import pandas as pd

def _obter_coluna(row, chaves_possiveis, padrao=0.0):
    """Função auxiliar para buscar colunas tolerando variações de nome (com/sem acento)."""
    for chave in chaves_possiveis:
        if chave in row:
            val = row[chave]
            if pd.notna(val):
                return val
    return padrao

def simular_demanda_e_caixa(df_catalogo, multiplicadores_preco, n_cenarios=2000, seed=42):
    """
    Estima demanda e caixa sob incerteza para diferentes multiplicadores de preço.
    Considera 3 fontes de variação: demanda, concorrente e custo logístico.
    """
    np.random.seed(seed)
    resultados = {}
    
    for idx, row in df_catalogo.iterrows():
        sku = row['sku']
        preco_base = _obter_coluna(row, ['preco_atual', 'preco atual'], 100.0)
        custo_unit = _obter_coluna(row, ['custo_unitario', 'custo unitario'], 50.0)
        custo_log = _obter_coluna(row, ['custo_logistico', 'custo logistico'], 5.0)
        
        demanda_90d = _obter_coluna(row, ['demanda_base_90d', 'demanda base 90d', 'demanda_base'], 300.0)
        demanda_base_sem = demanda_90d / 12.0
        
        elasticidade = abs(_obter_coluna(row, ['elasticidade'], 1.5))
        
        resultados[sku] = {}
        
        for m in multiplicadores_preco:
            preco_oferta = preco_base * m
            
            # Variação 1: Incerteza na demanda do consumidor (distribuição log-normal)
            ruido_demanda = np.random.lognormal(mean=0, sigma=0.15, size=n_cenarios)
            
            # Variação 2: Movimento de preço do concorrente (fator multiplicativo)
            fator_concorrente = np.random.uniform(0.9, 1.1, size=n_cenarios)
            
            # Variação 3: Incerteza no custo logístico (variação percentual)
            fator_custo_log = np.random.uniform(0.95, 1.25, size=n_cenarios)
            custo_log_sim = custo_log * fator_custo_log
            
            # Cálculo da demanda ajustada pela elasticidade e concorrente
            var_preco_pct = (preco_oferta - (preco_base * fator_concorrente)) / (preco_base * fator_concorrente)
            demanda_est = demanda_base_sem * (1 - elasticidade * var_preco_pct) * ruido_demanda
            demanda_est = np.maximum(0, demanda_est)
            
            # Lucro/Caixa gerado por unidade vendida
            margem_un = preco_oferta - custo_unit - custo_log_sim
            caixa_sim = demanda_est * margem_un
            
            resultados[sku][m] = {
                'demanda_media': np.mean(demanda_est),
                'caixa_medio': np.mean(caixa_sim),
                'demanda_std': np.std(demanda_est),
                'caixa_std': np.std(caixa_sim)
            }
            
    return resultados


def validar_plano_monte_carlo(df_catalogo, plano_precos, metas_semanais, n_cenarios=5000, seed=123):
    """
    Valida o plano de reprecificação gerado usando 5.000 cenários de Monte Carlo independentes.
    Verifica a probabilidade de honrar os compromissos de dívida nas semanas 4, 8 e 12.
    """
    np.random.seed(seed)
    n_semanas = 12
    caixa_acumulado_cenarios = np.zeros(n_cenarios)
    falhas = {4: 0, 8: 0, 12: 0}
    
    # Simulação horizonte de 12 semanas
    for sem in range(1, n_semanas + 1):
        caixa_semana = np.zeros(n_cenarios)
        for idx, row in df_catalogo.iterrows():
            sku = row['sku']
            mult_preco = plano_precos[sku][sem - 1]
            
            preco_base = _obter_coluna(row, ['preco_atual', 'preco atual'], 100.0)
            custo_unit = _obter_coluna(row, ['custo_unitario', 'custo unitario'], 50.0)
            custo_log = _obter_coluna(row, ['custo_logistico', 'custo logistico'], 5.0)
            demanda_90d = _obter_coluna(row, ['demanda_base_90d', 'demanda base 90d', 'demanda_base'], 300.0)
            elasticidade = abs(_obter_coluna(row, ['elasticidade'], 1.5))
            
            preco_oferta = preco_base * mult_preco
            custo_log_sim = custo_log * np.random.uniform(0.95, 1.25, size=n_cenarios)
            
            demanda_base_sem = demanda_90d / 12.0
            
            ruido_dem = np.random.lognormal(0, 0.15, size=n_cenarios)
            fat_conc = np.random.uniform(0.9, 1.1, size=n_cenarios)
            
            var_p = (preco_oferta - (preco_base * fat_conc)) / (preco_base * fat_conc)
            dem = np.maximum(0, demanda_base_sem * (1 - elasticidade * var_p) * ruido_dem)
            
            caixa_semana += dem * (preco_oferta - custo_unit - custo_log_sim)
            
        caixa_acumulado_cenarios += caixa_semana
        
        # Checagem das semanas de vencimento de dívida (4, 8, 12)
        if sem in metas_semanais:
            meta = metas_semanais[sem]
            falhas[sem] = np.sum(caixa_acumulado_cenarios < meta)

    prob_sucesso = {sem: 1.0 - (falhas[sem] / n_cenarios) for sem in metas_semanais}
    
    return {
        'caixa_medio_final': np.mean(caixa_acumulado_cenarios),
        'caixa_p5': np.percentile(caixa_acumulado_cenarios, 5),
        'caixa_p50': np.percentile(caixa_acumulado_cenarios, 50),
        'caixa_p95': np.percentile(caixa_acumulado_cenarios, 95),
        'probabilidades_pagamento': prob_sucesso,
        'cenarios_caixa': caixa_acumulado_cenarios
    }