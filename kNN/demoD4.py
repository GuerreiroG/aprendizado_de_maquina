"""
Autores: Daniel de Paula, Gustavo Guerreiro, Mayara Cardoso Simões
"""

import knn
import scipy.io as scipy

"""
Grupo de Dados 4:
"""

if __name__ == "__main__":
    # Importando dataset do grupo de dados 4
    mat = scipy.loadmat('grupoDados4.mat')
    grupoTest = mat['testSet']
    grupoTrain = mat['trainSet']
    testRots = mat['testLabs']
    trainRots = mat['trainLabs']

    # Q4.1: aplique seu algoritmo K-NN ao problema. Qual é a sua acurácia de classificação? ≈77% com k=4
    print('Calculando acurácia com os dados não normalizados:')
    knn.calcular_maior_acuracia(grupoTrain, trainRots, grupoTest, testRots)

    # Q4.2: A acurácia pode chegar a 92% com o K-NN. Descubra por que o resultado atual é muito menor. Ajuste o conjunto
    # de dados ou o valor de k de forma que a acurácia atinja 92% e explique o que você fez e por quê. Observe que,
    # desta vez, há mais de um problema...
    # R: Caso não se utilize a última coluna e se normalize os dados a acurácia é de 92% com k=7, porém, pode-se
    # alcançar uma acurácia ainda maior (97%) usando um k=14.

    # Normalizando os dados de teste e treino (sem a última coluna)
    grupo_test_normalizado = knn.normalizacao(grupoTest[:, :3])
    grupo_train_normalizado = knn.normalizacao(grupoTrain[:, :3])

    print('Calculando acurácia com os dados normalizados e desconsiderando a última coluna:')
    knn.calcular_maior_acuracia(grupo_train_normalizado, trainRots, grupo_test_normalizado, testRots)
