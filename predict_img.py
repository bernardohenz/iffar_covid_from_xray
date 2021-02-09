from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_name', required=True, help='especifique a imagem a ser analisada')
parser.add_argument('--model', default='./trained_model/trained_model.h5', help='especifique o path para o modelo treinado')
parser.add_argument('--num_tta', default=10, help='especifique o número de vezes que deve ser rodado o TTA (Test-time augmentation)')

opt = parser.parse_args()

# Criando gerador para conjunto de teste, dataaugs servem para TTA (Test-time augmentation)
test_datagen_tta = ImageDataGenerator(rescale=1./255,
                                      rotation_range=30,   # rotações aleatórias de até 40 graus no sentido horário ou antihorário
                                      width_shift_range=0.1,  # shift aleatórios horizontais de até 20% da largura da imagem
                                      height_shift_range=0.1, # shift aleatórios verticais de até 20% da altura da imagem
                                      shear_range=0.05, # shearing movendo vértices aleatoriamente até 20%
                                      zoom_range=0.05,  # zoom-out ou zoom-in com valores aleatórios até 20%
                                      horizontal_flip=True) # aleatoriamente espelhar horizontalmente as imagens


# Carregar best model
full_model = keras.models.load_model(opt.model)

imagePIL = keras.preprocessing.image.load_img(opt.img_name, target_size=(142, 142))
image = keras.preprocessing.image.img_to_array(imagePIL)
image = np.array([image])  

image_flow = test_datagen_tta.flow(image)

Y_pred = 0
for i in range(opt.num_tta):
    Y_pred = Y_pred + full_model.predict(image_flow)[0,0]
Y_pred = Y_pred/opt.num_tta

print('Esta imagem tem {:.2f}% de ter COVID.'.format( (1 - Y_pred)*100 ))  #  0 - covid     1 - normal