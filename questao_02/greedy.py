import pandas as pd

def _obter_coluna(row, chaves_possiveis, padrao=0.0):
    """Função auxiliar para buscar colunas tolerando variações de nome no CSV."""
    for chave in chaves_possiveis:
        if chave in row:
            val = row[chave]
            if pd.notna(val):
                return val
    return padrao

def alocar_promocoes_greedy(df_catalogo, plano_precos_dp, orcamento_semanal=500.0, n_semanas=12):
    """
    Algoritmo Greedy Semanal: Aloca um limite de orçamento promocional para os produtos
    com maior retorno/benefício marginal por real gasto.
    """
    plano_ajustado = {}
    for _, row in df_catalogo.iterrows():
        sku = _obter_coluna(row, ['sku', 'SKU', 'id_produto'], 'SKU_DESCONHECIDO')
        if sku in plano_precos_dp:
            trajetoria = list(plano_precos_dp[sku])
        else:
            trajetoria = []
            
        # Garante exatamente 12 semanas (preenchendo com o último valor ou 1.0 se faltar)
        if len(trajetoria) < n_semanas:
            ultimo_val = trajetoria[-1] if trajetoria else 1.0
            trajetoria.extend([ultimo_val] * (n_semanas - len(trajetoria)))
        elif len(trajetoria) > n_semanas:
            trajetoria = trajetoria[:n_semanas]
            
        plano_ajustado[sku] = trajetoria

    registro_alocacoes = []

    for sem in range(n_semanas):
        candidatos = []
        for idx, row in df_catalogo.iterrows():
            sku = _obter_coluna(row, ['sku', 'SKU', 'id_produto'], f'SKU_{idx}')
            mult_atual = plano_ajustado[sku][sem]
            
            preco_base = _obter_coluna(row, ['preco_atual', 'preco atual'], 100.0)
            custo_unit = _obter_coluna(row, ['custo_unitario', 'custo unitario'], 50.0)
            demanda_90d = _obter_coluna(row, ['demanda_base_90d', 'demanda base 90d', 'demanda_base'], 300.0)
            
            # Custo de aplicar destaque/frete promocional
            custo_promo = preco_base * 0.05
            
            # Ganho marginal estimado na demanda (+15% de incremento na demanda com a promoção)
            incremento_caixa = (preco_base * mult_atual - custo_unit) * (demanda_90d / 12.0) * 0.15
            
            roi_marginal = incremento_caixa / custo_promo if custo_promo > 0 else 0
            
            candidatos.append({
                'sku': sku,
                'custo': custo_promo,
                'ganho': incremento_caixa,
                'roi': roi_marginal,
                'mult_atual': mult_atual
            })
        
        # Ordenação gulosa pelo maior ROI marginal
        candidatos.sort(key=lambda x: x['roi'], reverse=True)
        
        orcamento_restante = orcamento_semanal
        for c in candidatos:
            if c['custo'] <= orcamento_restante and c['roi'] > 1.0:
                orcamento_restante -= c['custo']
                sku_atual = c['sku']
                plano_ajustado[sku_atual][sem] = round(c['mult_atual'] * 0.95, 2)
                
                registro_alocacoes.append({
                    'semana': sem + 1,
                    'sku': sku_atual,
                    'investimento': c['custo'],
                    'roi_estimado': c['roi']
                })

    return plano_ajustado, pd.DataFrame(registro_alocacoes)