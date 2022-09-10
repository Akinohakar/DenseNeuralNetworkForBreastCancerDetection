'''
Momento de retroalimentacion 2
Alan Eduardo Aquino Rosas

Para este modelo se trabajo sobre un dataset para la detencion de cancer de seno
Debido a la complejidad del dataset,con un total de 30 caracteristicas/dimensiones se decidio usar una metodo de la libreria de scikit-learn para redes neurnales.
Asi en las capas ocultas se puedan computar nuevas caracteristicas y funciones mas interesantes

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

La variable dependiente que queremos predecir correctamente a partir de nuestras caracteristicas es Si el tumor es benigno o maligno

Si es benigno y=0
Si es maligno y=1


Para la configuracion de la red neuronal se ocuparon 7 capas ocultas,por automatico una de salida

La funcion de activacion elegida fue la relu debido a que funciona mejor para redes neuronales
'''


#Importacion de librerias 
import pandas as pd #For dataset reading
from sklearn.model_selection import train_test_split #For Training/Cross-Validation/Test
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt #For graphing


#Preparacion de dataset
DF=pd.read_csv("./Dataset/data.csv")
DF["diagnosis"]=DF["diagnosis"].map({'M':0,'B':1})
df_y=DF["diagnosis"]
df_x=DF.drop("diagnosis",axis=1)
df_x=df_x.drop("Unnamed: 32",axis=1)

#Se escalan los datos para mejor convergencia 
scaler=preprocessing.StandardScaler().fit(df_x)
df_x=scaler.transform(df_x)

#Split entre training y dev set 
train_x,test_x,train_y,test_y=train_test_split(df_x,df_y,random_state=0)




#Configuracion de red neuronal para clasificacion
cancer_nn=MLPClassifier(random_state = 1,
                        hidden_layer_sizes = (20, 20, 20, 20,20,20,20),
                        activation = "relu",
                        verbose = False,
                        solver = "lbfgs",
                        learning_rate = "adaptive", 
                        max_iter = 20000)

#Entrenamiento red neuronal
cancer_nn.fit(train_x,train_y)

#Resultados red neuronal
print("Training Score",cancer_nn.score(train_x,train_y))
print("My test score",cancer_nn.score(test_x,test_y))

#Prediciones 
print("---------------------------Predicciones--------------------------------------")
for i in range(0,100,10):
    print("Cancer? Estimado",cancer_nn.predict(df_x[i,:].reshape(1,-1)),"Real",df_y[i])
