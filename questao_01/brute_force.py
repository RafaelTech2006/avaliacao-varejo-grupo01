import itertools 
import pandas as pd

class LeitorCsv:
    """Classe responsável pela leitura e extração de informações do CSV."""

    def __init__(self, caminho : str) -> None:
        """Inicializando a instância do objeto caminho."""
        self.caminho = caminho

    def leitura(self) -> pd.DataFrame:
        """Carregando o CSV e retornando 6 dados aleatórios do DataFrame."""
        return pd.read_csv(self.caminho).sample(6)

class PrecificadorSKU:
    """
        Classe responsável por aplicar o modelo matemático
        de precificação, elasticidade e validação de margem
        para um único SKU.
    """
    def __init__(self, piso_margem_minimo: float = 0.0831):
        self.piso_margem_minima = piso_margem_minimo

    def calcular_caixa_produto(self, preco_candidato: float, row: pd.Series) -> float:
        """Calculando o caixa gerado para um SKU dado um preço candidato"""

        # Variação percentual do preço em relação ao preço atual
        var_preco = (preco_candidato - row["preco_atual"]) / row["preco_atual"]

        # Elasticidade e variação da demanda
        var_demanda = -row["elasticidade"] * var_preco
        demanda_novo = row["demanda_base_90d"] * (1 + var_demanda)
        demanda_novo = max(0, demanda_novo) # Demanda não pode ser negativa

        # Trava de estoque
        qtd_vendida = min(demanda_novo, row["estoque"])

        # Receita líquida e custo total unitário
        receita_liquida = preco_candidato * (1 - row["taxa_marketplace"])
        custo_total_unitario = row["custo_unitario"] + row["custo_logistico"]
        caixa_unitario = receita_liquida - custo_total_unitario

        # Validação do piso de margem 
        margem_unitaria = (
            caixa_unitario / preco_candidato if preco_candidato > 0 else 0
        )
        if margem_unitaria < self.piso_margem_minima:
            return -float("inf") # Combinação inviável por violação de margem

        # Caixa total gerado pelo SKU
        return qtd_vendida * caixa_unitario

class BruteForce:
    """TODO"""

            
    





   


    


    


