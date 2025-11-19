import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import glob
import os

PASTA_NOVAS_IMAGENS = "TestesExtras"

CAMINHO_MODELO = "cats_dogs_model_v3.keras"

TAMANHO_IMAGEM = 224

CLASS_NAMES = ["Cat", "Dog"]

def prever_novas_imagens():

    print(f"Carregando modelo de {CAMINHO_MODELO}...")
    try:
        modelo = tf.keras.models.load_model(CAMINHO_MODELO, compile=False)
    except Exception as e:
        print(f"\n--- ERRO ---")
        print(f"Não foi possível carregar o modelo: {e}")
        print("------------\n")
        return

    print("Modelo carregado com sucesso.")

    if not os.path.exists(PASTA_NOVAS_IMAGENS):
        os.makedirs(PASTA_NOVAS_IMAGENS)
        print(f"\nPasta '{PASTA_NOVAS_IMAGENS}' criada.")
        print(f"Rode o script novamente.")
        return

    tipos_imagem = ('*.jpg', '*.jpeg')
    caminhos_imagens = []
    for tipo in tipos_imagem:
        caminhos_imagens.extend(glob.glob(os.path.join(PASTA_NOVAS_IMAGENS, tipo)))

    if not caminhos_imagens:
        print(f"\nNenhuma imagem (*.jpg, *.jpeg, *.png) encontrada em '{PASTA_NOVAS_IMAGENS}'.")
        return

    print(f"\nEncontradas {len(caminhos_imagens)} imagens. Iniciando previsões...")
    print("-" * 40)

    for caminho in caminhos_imagens:
        try:
            img = image.load_img(caminho, target_size=(TAMANHO_IMAGEM, TAMANHO_IMAGEM))

            img_array = image.img_to_array(img)

            img_batch = np.expand_dims(img_array, axis=0)

            # d. Aplicar o pré-processamento EXATO da MobileNetV2
            img_preprocessed = preprocess_input(img_batch)

            # e. Fazer a predição
            prediction = modelo.predict(img_preprocessed, verbose=0)

            # f. Interpretar o resultado (saída sigmoid)
            probabilidade = prediction[0][0]

            if probabilidade > 0.5:
                classe_id = 1  # Dog
            else:
                classe_id = 0  # Cat

            nome_classe = CLASS_NAMES[classe_id]
            nome_arquivo = os.path.basename(caminho)

            if classe_id == 0:
                prob = (1 - probabilidade)
            else:
                prob = probabilidade

            print(f"Arquivo: {nome_arquivo.ljust(30)} -> Classe: {nome_classe} ({round(prob,2)*100}%)")

        except Exception as e:
            print(f"Erro ao processar {os.path.basename(caminho)}: {e}")

    print("-" * 40)
    print("Previsões concluídas.")


if __name__ == "__main__":
    prever_novas_imagens()