# Dense Neural Network For Breast Cancer Detection

Trabajo para:
Momento de retroalimentacion 2


Hecho por:
Alan Eduardo Aquino Rosas

## Arquitectura del modelo

Para este modelo se trabajo sobre un dataset para la detencion de cancer de seno
Debido a la complejidad del dataset,con un total de 30 caracteristicas/dimensiones se decidio usar una metodo de la libreria de scikit-learn para redes neurnales.
Asi en las capas ocultas se puedan computar nuevas caracteristicas y funciones mas interesantes

El tipo de red neurnal de scikit learn que se ocupo es MLPClassifier

Para la configuracion de la red neuronal se ocuparon 7 capas ocultas,cada capa con 20 neuronas y por automatico una de salida(y_gorrito)
La funcion de activacion elegida fue la relu debido a que funciona mejor para redes neuronales

## Caracteristicas del dataset

Dataset de tumores benignos y malignos de Seno

Link:

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

### Variables independientes
Las 10 principales  caracteristicas del dataset son:


a) radius (mean of distances from center to points on the perimeter)


b) texture (standard deviation of gray-scale values)


c) perimeter


d) area


e) smoothness (local variation in radius lengths)


f) compactness (perimeter^2 / area - 1.0)


g) concavity (severity of concave portions of the contour)


h) concave points (number of concave portions of the contour)


i) symmetry


j) fractal dimension ("coastline approximation" - 1)



Las demas caracteristicas son medidas estadisticas


### Variable dependiente
La variable dependiente que queremos predecir correctamente a partir de nuestras caracteristicas es Si el tumor es benigno o maligno

Si es benigno y=0
Si es maligno y=1

### Preprocesamiento
Debido a que las escalas de las caracterisitcas eran muy altas y variaban mucho se decidio escalar todas las caracteristicas mediante el StandarScaler de Scikit-Learn encontrado en preprocessing para que el modelo funcione mejor y converga mas rapido



Igualmente,se dividio el dataset en training y validation set.



El training para entrenar el modelo y el validation set para verificar que generalize bien y no este memorizando.

## Resultados
En el training set obtenemos un score de 1 representado que el 100% se predijo correctamente 


En el validation set tenemos un score de 0.958 representando que del 100 se predijo el 95.8% correctamente 
