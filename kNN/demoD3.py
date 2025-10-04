"""
Autores: Daniel de Paula, Gustavo Guerreiro, Mayara Cardoso Simões
"""

import knn
import scipy.io as scipy

"""
Grupo de Dados 3:
"""

if __name__ == "__main__":
    # Importando dataset do grupo de dados 3
    mat = scipy.loadmat('grupoDados3.mat')
    grupoTest = mat['grupoTest']
    grupoTrain = mat['grupoTrain']
    testRots = mat['testRots']
    trainRots = mat['trainRots']

    # Q3.1: aplique o kNN ao problema usando k = 1. Qual é a acurácia na classificação?
    # R: 62% com k=1
    acuracia_calculada = knn.calcular_acuracia(knn.meu_knn(grupoTrain, trainRots, grupoTest, 1), testRots)
    print(f'Acurácia com k=1: {round(acuracia_calculada*100)}%')

    # Q3.2: A acurácia pode ser igual a 92% com o kNN. Descubra por que o resultado atual é muito menor. Ajuste o
    # conjunto de dados ou k de tal forma que a acurácia se torne 92% e explique o que você fez e por quê.
    # R: Foi necessário normalizar os valores, após buscar pela maior acurácia, conseguiu-se chegar ao valor de 92% com
    # k=14, mas foi possível chegar a 94% com k=25. Isso ocorreu, pois os dados estavam em escalas muito diferentes, o
    # que influenciava na distância entre os pontos.

    # Normalizando os dados de teste e treino
    grupo_test_normalizado = knn.normalizacao(grupoTest)
    grupo_train_normalizado = knn.normalizacao(grupoTrain)

    print('Acurácia com os dados normalizados:')
    knn.calcular_maior_acuracia(grupo_train_normalizado, trainRots, grupo_test_normalizado, testRots)