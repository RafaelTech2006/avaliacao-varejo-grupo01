import numpy as np
import pandas as pd

def _obter_coluna(row, chaves_possiveis, padrao=0.0):
    """Função auxiliar para buscar colunas tolerando variações de nome no CSV."""
    for chave in chaves_possiveis:
        if chave in row:
            val = row[chave]
            if pd.notna(val):
                return val
    return padrao

class OtimizadorDP:
    """
    Programação Dinâmica para encontrar a trajetória de preços de 12 semanas.
    Estado: (semana, estoque_restante_discretizado)
    Decisão: multiplicador de preço a aplicar no produto.
    """
    def __init__(self, sku_data, estatisticas_mc, multiplicadores, n_semanas=12):
        self.sku = _obter_coluna(sku_data, ['sku', 'SKU', 'id_produto'], 'SKU_DESCONHECIDO')
        estoque_val = _obter_coluna(sku_data, ['estoque', 'estoque_inicial', 'quantidade_estoque', 'qtd_estoque'], 100.0)
        self.estoque_inicial = int(float(estoque_val))
        
        self.est_mc = estatisticas_mc[self.sku]
        self.multiplicadores = multiplicadores
        self.n_semanas = n_semanas
        self.memo = {}

    def resolver(self):
        # Passo do grid de estoque para reduzir espaço de estados se o estoque for grande
        step_estoque = max(1, self.estoque_inicial // 50)
        
        def dp(semana, estoque):
            if semana > self.n_semanas or estoque <= 0:
                return 0.0, []
            
            # Discretização do estado de estoque para memoization
            est_key = (estoque // step_estoque) * step_estoque
            state = (semana, est_key)
            
            if state in self.memo:
                return self.memo[state]
            
            melhor_caixa = -float('inf')
            melhor_caminho = []
            
            for m in self.multiplicadores:
                demanda_est = self.est_mc[m]['demanda_media']
                caixa_est = self.est_mc[m]['caixa_medio']
                
                vendas = min(estoque, demanda_est)
                fator_venda = vendas / demanda_est if demanda_est > 0 else 1.0
                caixa_gerado = caixa_est * fator_venda
                
                prox_estoque = max(0, int(estoque - vendas))
                
                caixa_futuro, caminho_futuro = dp(semana + 1, prox_estoque)
                caixa_total = caixa_gerado + caixa_futuro
                
                if caixa_total > melhor_caixa:
                    melhor_caixa = caixa_total
                    melhor_caminho = [m] + caminho_futuro
            
            self.memo[state] = (melhor_caixa, melhor_caminho)
            return self.memo[state]

        caixa_opt, trajetória = dp(1, self.estoque_inicial)
        return caixa_opt, trajetória

def otimizar_catalogo_dp(df_catalogo, estatisticas_mc, multiplicadores):
    plano_precos = {}
    caixa_esperado_total = 0.0
    
    for idx, row in df_catalogo.iterrows():
        sku = _obter_coluna(row, ['sku', 'SKU', 'id_produto'], f'SKU_{idx}')
        dp_solver = OtimizadorDP(row, estatisticas_mc, multiplicadores)
        caixa_opt, trajetoria = dp_solver.resolver()
        plano_precos[sku] = trajetoria
        caixa_esperado_total += caixa_opt
        
    return plano_precos, caixa_esperado_total