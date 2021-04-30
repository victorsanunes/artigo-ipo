# Gerando de Conjuntos de Dados de classificação
# Aron Ifanger Maciel (aronifanger@gmail.com)
# Ana Carolina Lorena (aclorena@gmail.com)

import pandas as pd

from gera_entradaClassificacao import gera_entrada_classificacao
from gera_saidaClassificacao import gera_saida_classificacao

def gera_conjunto_classificacao(
    nome,
    quantidade_de_linhas,
    quantidade_de_colunas,
    funcao,
    erro
):
    x = gera_entrada_classificacao(
        quantidade_de_linhas, 
        quantidade_de_colunas
    )
    saida = gera_saida_classificacao(x, erro, funcao)
    df = pd.DataFrame(
        x,
        columns=['V{}'.format(i+1) for i in range(x.shape[1])]
    )
    df['y'] = saida['saida']
    # df.to_csv(nome)
    return df
