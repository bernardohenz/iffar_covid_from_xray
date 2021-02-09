from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from util import plot_confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix


test_dir = './dataset/test/'
# Criando gerador para conjunto de teste, dataaugs servem para TTA (Test-time augmentation)
test_datagen_tta = ImageDataGenerator(rescale=1./255,
                                      rotation_range=30,   # rotações aleatórias de até 40 graus no sentido horário ou antihorário
                                      width_shift_range=0.1,  # shift aleatórios horizontais de até 20% da largura da imagem
                                      height_shift_range=0.1, # shift aleatórios verticais de até 20% da altura da imagem
                                      shear_range=0.05, # shearing movendo vértices aleatoriamente até 20%
                                      zoom_range=0.05,  # zoom-out ou zoom-in com valores aleatórios até 20%
                                      horizontal_flip=True) # aleatoriamente espelhar horizontalmente as imagens

test_generator_tta = test_datagen_tta.flow_from_directory(test_dir,
                                                          target_size=(142, 142),
                                                          batch_size=20,
                                                          class_mode='binary',
                                                          shuffle = False)


# Carregar best model
full_model = keras.models.load_model('./trained_model/trained_model.h5')

num_ttas = 10
Y_pred = np.zeros((len(test_generator_tta.classes),))
for i in range(num_ttas):
  Y_pred = Y_pred+full_model.predict(test_generator_tta)[:,0]
Y_pred = Y_pred/num_ttas
Y_pred = np.round(Y_pred)

conf_matrix = confusion_matrix(test_generator_tta.classes, Y_pred)

acc_tta = sklearn.metrics.accuracy_score(test_generator_tta.classes, Y_pred)
print('Acurácia no conjunto de teste: {}'.format(acc_tta))
f1_score = sklearn.metrics.f1_score(test_generator_tta.classes, Y_pred)
print('F1-score no conjunto de teste: {}'.format(f1_score))


plot_confusion_matrix(test_generator_tta.classes,Y_pred,  classes=list(test_generator_tta.class_indices.keys()), normalize=True,
                      title='COVID-19 Detection - TTA', output_name='covid_detect_conf_matrix')