"""
Autores: Daniel de Paula, Gustavo Guerreiro, Mayara Cardoso Simões
"""

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import scipy.io as scipy

def d(p, q):
    """
    Calcula a distância Euclidiana entre dois pontos 'p' e 'q'.
    """
    soma = 0
    for i in range(len(p)):
        soma += (p[i] - q[i])**2
    return np.sqrt(soma)


def dist(dados_treino, dados_teste):
    """
    Calcula a matriz de distâncias entre os dados de treino e os dados de teste.
    """
    matriz_distancia = np.zeros((len(dados_teste), len(dados_treino)))
    for i in range(len(dados_teste)):
        for j in range(len(dados_treino)):
            distancia = d(dados_teste[i], dados_treino[j])
            matriz_distancia[i, j] = distancia
    return matriz_distancia


def meu_knn(dados_treino, rotulo_treino, dados_teste, k):
    """
    Implementa o algoritmo k-NN.
    1. Calcula a distância entre o exemplo de teste e os dados de treinamento
    2. Ordena as distâncias. A ordem iX de cada elemento ordenado
    3. Obter os rótulos correspondentes aos exemplos mais próximos k
    4. A moda dos rótulos correspondentes são os rótulos previstos
    5. Retorna os rótulos previstos
    """
    matriz_distancia = dist(dados_treino, dados_teste)
    indices_ordenados = [np.argsort(linha) for linha in matriz_distancia]
    indices_rotulados = [[rotulo_treino[indice] for indice in indices] for indices in indices_ordenados]
    y_circunflexo = [mode(rotulos[:k])[0][0] for rotulos in indices_rotulados]
    return y_circunflexo


def calcular_acuracia(rotulos_previsao, rotulos_teste):
    """
    Calcula a acurácia da classificação.
    """
    num_correto = 0
    for previsto, teste in zip(rotulos_previsao, rotulos_teste):
        if previsto == teste:
            num_correto += 1
    total_num = len(rotulos_teste)
    return num_correto / total_num


def calcular_maior_acuracia(grupo_train, train_rots, grupo_test, test_rots):
    """
    Calcula a maior acurácia possível variando o valor de k.
    Testando valores de k de 1 até a metade do número de exemplos de treino.
    Imprime a acurácia para cada valor de k e o maior valor encontrado.
    """
    maior_acuracia = -float('inf')
    maior_k = -1
    for i in range(1, (len(grupo_train)+1)//2):
        rotulos_previstos = meu_knn(grupo_train, train_rots, grupo_test, i)
        acuracia = calcular_acuracia(rotulos_previstos, test_rots)
        print(f'k={i} - Acurácia: {round(acuracia*100)}%')
        if acuracia > maior_acuracia:
            maior_acuracia = acuracia
            maior_k = i
    print(f'Maior acurácia: {round(maior_acuracia*100)}%')
    print(f'k={maior_k}')


def obter_dados_rotulo(dados, rotulos, rotulo, indice):
    """
    Retorna os dados de um determinado rótulo contido na lista de rótulos.
    """
    ret = []
    for idx in range(0, len(dados)):
        if rotulos[idx] == rotulo:
            ret.append(dados[idx][indice])
    return ret


def visualiza_pontos(dados, rotulos, d1, d2, titulo=''):
    """
    Visualiza os pontos em um gráfico de dispersão (scatter plot).
    Cada rótulo é representado por uma cor e um marcador diferente.
    """
    plt.scatter(obter_dados_rotulo(dados, rotulos, 1, d1), obter_dados_rotulo(dados, rotulos, 1, d2), c='red', marker='^')
    plt.scatter(obter_dados_rotulo(dados, rotulos, 2, d1), obter_dados_rotulo(dados, rotulos, 2, d2), c='blue', marker='+')
    plt.scatter(obter_dados_rotulo(dados, rotulos, 3, d1), obter_dados_rotulo(dados, rotulos, 3, d2), c='green', marker='.')
    plt.title(titulo)
    plt.show()
    plt.clf()


def normalizar_valores(valores):
    """
    Normaliza os valores para o intervalo [0, 1].
    """
    minimo = min(valores)
    maximo = max(valores)
    return  [(valor-minimo)/(maximo-minimo) for valor in valores]


def transpor_matriz(matriz):
    """
    Transpõe uma matriz.
    """
    return [coluna for coluna in zip(*matriz)]


def normalizacao(grupo):
    """
    Normaliza os dados de um grupo usando a normalização dos valores e a transposição da matriz.
    """
    return transpor_matriz([normalizar_valores(dados) for dados in zip(*grupo)])


# Testes indicados no enunciado:
if __name__ == '__main__':
    # Para testar se você implementou a função corretamente, baixe o arquivo **grupoDados1.mat** - cada arquivo **.mat**
    # contém 4 variáveis que são: **grupoTest, grupoTrain, testRots, trainRots**. Para baixar os arquivos **.mat**
    # no Python você pode fazer o seguinte:

    mat = scipy.loadmat('grupoDados1.mat')
    grupoTest = mat['grupoTest']
    grupoTrain = mat['grupoTrain']
    testRots = mat['testRots']
    trainRots = mat['trainRots']

    # Então, verifique quantas classes foram previstas corretamente, isto é chamado de *acurácia (accuracy)*:
    print('Calculando Acurácia:')
    acuracia = calcular_acuracia(meu_knn(grupoTrain, trainRots, grupoTest, 1), testRots)
    print(f"{int(acuracia*100)}%")

    # A acurácia deve ser de 96%. Agora, vamos estender a função a um classificador k-NN:
    # Para cada exemplo de teste
    # Calcule a distância entre o exemplo de teste e os dados de treinamento
    # Ordene as distâncias. A ordem iX de cada elemento ordenado é importante:
    # [distOrdenada ind] = sort(...);
    # Obter os rótulos correspondentes aos exemplos mais próximos k
    # Agora, a moda dos rótulos correspondentes são os rótulos previstos (você pode usar a função mode).

    # Teste novamente no conjunto de dados 1 (**grupoDados1.mat**) e utilize k = 10 para uma acurácia igual a 94%.
    rotulos_previstos = meu_knn(grupoTrain, trainRots, grupoTest, 10)
    acuracia = calcular_acuracia(rotulos_previstos, testRots)
    print(f"{int(acuracia*100)}%")
    # É sempre bom visualizar graficamente os seus dados.
    visualiza_pontos(grupoTest, rotulos_previstos, 1, 2, 'Rótulos Previstos')
    visualiza_pontos(grupoTest, testRots, 1, 2, 'Rótulos Corretos:')
