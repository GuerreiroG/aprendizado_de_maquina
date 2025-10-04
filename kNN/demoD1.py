"""
Autores: Daniel de Paula, Gustavo Guerreiro, Mayara Cardoso Simões
"""

import knn
import scipy.io as scipy

"""
Grupo de Dados 1:
O grupoDados1 é um conjunto de dados de flores. Para mais informações consulte: 
http://archive.ics.uci.edu/ml/datasets/Iris .
"""


if __name__ == "__main__":
    # Importando dataset do grupo de dados 1
    mat = scipy.loadmat('grupoDados1.mat')
    grupoTest = mat['grupoTest']
    grupoTrain = mat['grupoTrain']
    testRots = mat['testRots']
    trainRots = mat['trainRots']

    # Q1.1. Qual é a acurácia máxima que você consegue da classificação? 98% usando k=3
    knn.calcular_maior_acuracia(grupoTrain, trainRots, grupoTest, testRots)

    # Q1.2. É necessário ter todas as características (atributos) para obter a acurácia máxima para esta classificação?
    # É possível chegar à mesma acurácia de 98% com k=4 apenas com os comprimentos das pétalas e das sépalas, as
    # larguras não são necessárias.
    print("Acurácia sem 'petal width':")
    knn.calcular_maior_acuracia(grupoTrain[:, :3], trainRots, grupoTest[:, :3], testRots)

    print("Acurácia sem 'petal width' e 'petal length':")
    knn.calcular_maior_acuracia(grupoTrain[:, :2], trainRots, grupoTest[:, :2], testRots)

    print("Acurácia sem 'sepal width' e 'petal width':")
    knn.calcular_maior_acuracia(grupoTrain[:, ::2], trainRots, grupoTest[:, ::2], testRots)
