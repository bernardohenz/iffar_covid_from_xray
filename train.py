import os
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix


from util import plot_confusion_matrix


# Definindo os diretórios de train/val/test
train_dir = './dataset/train/'
val_dir = './dataset/val/'
test_dir = './dataset/test/'


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Inserindo data augmentation no nosso ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=60,   # rotações aleatórias de até 40 graus no sentido horário ou antihorário
                                   width_shift_range=0.2,  # shift aleatórios horizontais de até 20% da largura da imagem
                                   height_shift_range=0.2, # shift aleatórios verticais de até 20% da altura da imagem
                                   shear_range=0.2, # shearing movendo vértices aleatoriamente até 20%
                                   zoom_range=0.2,  # zoom-out ou zoom-in com valores aleatórios até 20%
                                   horizontal_flip=True) # aleatoriamente espelhar horizontalmente as imagens

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(142, 142),
                                                    batch_size=20,
                                                    class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(142, 142),
                                                batch_size=20,
                                                class_mode='binary')
# Notem que somente aplicamos as transformações no conjunto de treino
# Conjuntos de validação e teste não necessitam de data-augmentation


# Carregando modelo para finetunning
conv_model = keras.applications.VGG16(include_top=False,
                                      weights='imagenet',
                                      input_shape=(142,142,3))

conv_model.trainable = True
for layer in conv_model.layers:   # Percorre todas as Layers e só deixa treinável as camadas do bloco 'block5_conv1'
  if 'block5' in layer.name:
    layer.trainable = True
  else:
    layer.trainable = False

full_model = keras.models.Sequential()
full_model.add(conv_model)
full_model.add(keras.layers.Flatten())
full_model.add(keras.layers.Dense(128, activation='relu'))
full_model.add(keras.layers.BatchNormalization())
full_model.add(keras.layers.Dropout(0.1))
full_model.add(keras.layers.Dense(1, activation='sigmoid'))


# Compilando modelo (definir função de loss e otimizador)
full_model.compile(loss=keras.losses.BinaryCrossentropy(),
                   optimizer=keras.optimizers.Adam(lr=1e-5),
                   metrics=keras.metrics.BinaryAccuracy())

# Criando callback que sempre verifica a loss de validação, e salva a que tem melhor/menor loss
mcp_save = keras.callbacks.ModelCheckpoint('./best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
history = full_model.fit(train_generator,
                         epochs=30,
                         validation_data=val_generator,
                         callbacks=[mcp_save])


# Plotar curvas de loss e acurácia para treino e validação
train_acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'bo', label='Acc de treino')
plt.plot(epochs, val_acc, 'b', label='Acc de validação')
plt.title('Acurácia de treino e validação - Treinamento de Fine-tunning')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Loss de treino')
plt.plot(epochs, val_loss, 'b', label='Loss de validação')
plt.title('Loss de treino e validação - Treinamento de Fine-tunning')
plt.legend()
plt.savefig('training_history.svg')



