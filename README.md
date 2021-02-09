# Deep-learning method for detecting COVID-19 from x-ray images




Este projeto foi desenvolvido com o intuito de auxiliar no combate da pandemia COVID-19.

A principal ideia do sistema desenvolvido é receber uma imagem de raio-x do pulmão, e analisar a probabilidade do paciente estar com COVID-19, auxiliando no diagnóstico médico.

Utilizar esse sistema **não substitui o diagnóstico médico**, uma vez que não é possível garantir 100% de precisão na detecção da doença.

## Dataset e Modelo treinado
Rode os códigos [download_dataset.py](download_dataset.py) e [download_trained_model.py](download_trained_model.py) para fazer o download do dataset e modelo pré-treinado, respectivamente. Nosso dataset foi construído a partir do [COVID-19 image data collection](https://github.com/ieee8023/covid-chestxray-dataset).

## Códigos 

* [train.py](train.py) apresenta o código utilizado para o treinamento do modelo.

* [test.py](test.py) avalia o modelo treinado no conjunto de teste, calculando a acurácia, f1-score e matriz de confusão.

* [predict_img.py](predict_img.py) apresenta o código para realizar a inferência de uma imagem. Use como exemplo ```python predict_img.py --img_name ./sample_imgs/covid.jpeg``` ou ```python predict_img.py --img_name ./sample_imgs/normal.png``` para rodar o código nas imagens de exemplo.



Todos os direitos são reservados aos autores deste software e ao Instituto Federal Farroupilha.