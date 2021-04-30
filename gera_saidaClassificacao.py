# Gerador de Conjuntos de Dados de classificação
# Aron Ifanger Maciel (aronifanger@gmail.com)
# Ana Carolina Lorena (aclorena@gmail.com)

import numpy as np

# Função para gerar a saída

def gera_saida_classificacao(
    x,
    erro=0.1,
    funcao='pol1'
):
    quantidade_de_linhas  = x.shape[0]
    quantidade_de_colunas = x.shape[1]
    qd = quantidade_de_colunas - 1
    xqd = x[:,:qd]

    if funcao == 'pol1':
        a  = 4.0 * np.random.rand(1, qd) - 2.0
        x1 = 2.0 * np.random.rand(1, qd) - 1.0
        y  = np.sum(a * (xqd - x1), axis=1)
    elif funcao == 'pol2':
        a  = 4.0 * np.random.rand(1, qd) - 2.0
        x1 = (2.0 * np.random.rand(1, qd) - 1.0) * 0.7
        x2 = (2.0 * np.random.rand(1, qd) - 1.0) * 1.4
        y  = np.sum(a * (xqd - x1) * (xqd - x2), axis=1)
    elif funcao == 'pol3':
        a  = 4.0 * np.random.rand(1, qd) - 2.0
        x1 = (2.0 * np.random.rand(1, qd) - 1.0) * 0.5
        x2 = (2.0 * np.random.rand(1, qd) - 1.0) * 1.0
        x3 = (2.0 * np.random.rand(1, qd) - 1.0) * 1.5
        y  = np.sum(a * (xqd - x1) * (xqd - x2) * (xqd - x3), axis=1)
    elif funcao == 'pol4':
        a  = 4.0 * np.random.rand(1, qd) - 2.0
        x1 = (2.0 * np.random.rand(1, qd) - 1.0) * 0.4
        x2 = (2.0 * np.random.rand(1, qd) - 1.0) * 0.8
        x3 = (2.0 * np.random.rand(1, qd) - 1.0) * 1.2
        x4 = (2.0 * np.random.rand(1, qd) - 1.0) * 1.6
        y  = np.sum(a * (xqd - x1) * (xqd - x2) * (xqd - x3) * (xqd - x4), axis=1)
    elif funcao == 'pol5':
        a  = 4.0 * np.random.rand(1, qd) - 2.0
        x1 = (2.0 * np.random.rand(1, qd) - 1.0) * 0.3
        x2 = (2.0 * np.random.rand(1, qd) - 1.0) * 0.7
        x3 = (2.0 * np.random.rand(1, qd) - 1.0) * 1.1
        x4 = (2.0 * np.random.rand(1, qd) - 1.0) * 1.4
        x5 = (2.0 * np.random.rand(1, qd) - 1.0) * 1.7
        y  = np.sum(a * (xqd - x1) * (xqd - x2) * (xqd - x3) * (xqd - x4) * (xqd - x5), axis=1)
    elif funcao == 'sin1':
        fase = np.random.rand(1, qd) * 2.0 * np.pi
        y    = np.sum(
            np.random.rand(1, qd) * np.sin(xqd * 2 * np.pi + fase), 
            axis=1
        )
    elif funcao == 'sin2':
        fase = np.random.rand(1, qd) * 2.0 * np.pi
        y    = np.sum(
            np.random.rand(1, qd) * np.sin(xqd * 4 * np.pi + fase), 
            axis=1
        )
    elif funcao == 'sin3':
        fase = np.random.rand(1, qd) * 2.0 * np.pi
        y    = np.sum(
            np.random.rand(1, qd) * np.sin(xqd * 6 * np.pi + fase), 
            axis=1
        )

    y += np.random.normal(scale=erro, size=quantidade_de_linhas)
    y = (x[:,qd] > y).astype(int).reshape(-1, 1)
    return {
        'saida': y
    }