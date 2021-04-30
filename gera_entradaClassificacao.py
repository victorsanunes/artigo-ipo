# Gerador de Conjuntos de Dados de classificação
# Aron Ifanger Maciel (aronifanger@gmail.com) 
# Ana Carolina Lorena (aclorena@gmail.com)

import numpy as np

# Função para gerar a entrada

def gera_entrada_classificacao(
    quantidade_de_linhas=500, 
    quantidade_de_colunas=1
):

    return 2.0 * np.random.rand(
        quantidade_de_linhas, 
        quantidade_de_colunas
    ) - 1.0

