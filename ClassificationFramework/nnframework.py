


#Importancion de librerias 
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
