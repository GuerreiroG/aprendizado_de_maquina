"""
Autores: Daniel de Paula, Gustavo Guerreiro, Mayara Cardoso Simões
"""

import knn
import scipy.io as scipy

"""
Grupo de Dados 2:
O Grupo de Dados 2 é um problema que visa prever a origem do vinho em base aos seus componentes químicos. 
As características são:

1) Álcool
2) Ácido málico
3) Cinzas
4) Alcalinidade das cinzas
5) Magnésio
6) Fenóis totais
7) Flavonóides
8) Fenóis não flavonóides
9) Proantocianinas
10) Intensidade de cor
11) Tonalidade
12) OD280 / OD315 de vinhos diluídos
13) Prolina
"""

if __name__ == "__main__":
    # Importando dataset do grupo de dados 2
    mat = scipy.loadmat('grupoDados2.mat')
    grupoTest = mat['grupoTest']
    grupoTrain = mat['grupoTrain']
    testRots = mat['testRots']
    trainRots = mat['trainRots']

    # Q2.1: aplique seu kNN a este problema. Qual é a sua acurácia de classificação?
    # R: 78% com k=10
    print('Calculando Acurácia com os dados não normalizados:')
    knn.calcular_maior_acuracia(grupoTrain, trainRots, grupoTest, testRots)

    # Q2.2: A acurácia pode ser igual a 98% com o kNN. Descubra por que o resultado atual é muito menor.
    # Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne 98% e explique o que você fez e por quê.
    # R: Após serem normalizados a acurácia pode ser 98% (k=1) ou ainda chegar até 100% com (k = 4)

    # Normalizando os dados de teste e treino
    grupo_test_normalizado = knn.normalizacao(grupoTest)
    grupo_train_normalizado = knn.normalizacao(grupoTrain)

    print('Calculando Acurácia com os dados normalizados:')
    knn.calcular_maior_acuracia(grupo_train_normalizado, trainRots, grupo_test_normalizado, testRots)

