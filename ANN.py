import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
import numpy as np
import pickle
data=pd.read_csv("/Users/nomanmacbook/Downloads/churn_modeling_sample.csv",sep=",")
print(data.head())
# data processing
#drop the irralevent colums 
data=data.drop(["RowNumber","CustomerId","Surname"], axis=1)
print(data.head())
#Encode catogarical variable
LEG=LabelEncoder()
data["Gender"]=LEG.fit_transform(data["Gender"])
print(data)
#OHE Geography
OHE=OneHotEncoder()
geo_encoders=OHE.fit_transform(data[["Geography"]])
#.toarray  Sparse matrix ko normal array mein convert karta hai
geo_encoded=pd.DataFrame(geo_encoders.toarray(),columns=OHE.get_feature_names_out(["Geography"]))
print(geo_encoded)
# combine All the colum with the original data
#axis=0 → rows par kaam karo
#axis=1 → columns par kaam karo
data=pd.concat([data.drop("Geography",axis=1),geo_encoded],axis=1)
print(data)
#save the encoder and OHE on pickile file

with open("LEG.pkl","wb") as file:
    pickle.dump(LEG,file)

with open("OHE.pkl","wb") as file:
    pickle.dump(OHE,file)
print(data.head())
# devide the dataset into dependent or independent set
x=data.drop("Exited",axis=1)
y=data["Exited"]
#split the data training and testing set\
#random_state=42 ka matlab: random kaam repeatable ho jaye
#Har dafa same result mile — split ya algorithm different behave na kare
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#Scale this feature
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.fit_transform(x_test)
print(x_train)

#save the Scalar on pickile file
with open("scalar.pkl","wb") as file:
    pickle.dump(scalar,file)
print(data)
#ANN EMPLEMENTATION
#TensorFlow ek open-source framework hai
# jo neural networks aur deep learning models banane ke liye popular hai
import tensorflow as tf
#Sequential model ek stack hai jismein layers ek ke baad ek add hote hain
# Ye ANN (Artificial Neural Network) banane ke liye asaan hai.
from tensorflow.keras.models import Sequential
#Dense layer mein har neuron previous layer ke har neuron se connected hota hai
# Ye fully connected layer kehlati hai.
from tensorflow.keras.layers import Dense
#EarlyStopping → Agar model ka loss improve na ho to training rok deta hai.
#TensorBoard: Training ke metrics (jaise loss, accuracy)
# ko visualize karta hai graphs ke through.
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
#datetime → Folder ka unique name banane ke liye time-date use kar rahe hain.
import datetime 
#biul our ANN model
#Sequential → Model line by line layers add karne ka tareeqa.
models=Sequential([
     Dense(64,activation="relu",input_shape=(x_train.shape[1],)),# 1st hidden layer
     Dense(32,activation="relu"),# 2nd hidden layer
     Dense(1,activation="sigmoid")# output layer
])
print(models.summary())
#Adam:Ek advanced optimization algorithm hai jo weights ko update karta hai.
#learning_rate=0.01: model kis speed se seekhega.
#loss = binary_crossentropy: kyunki hum 0/1 (true/false) predict kar rahe hain.
#metrics=["accuracy"]: training ke waqt accuracy dikhaye.
opt=tf.keras.optimizers.Adam(learning_rate=0.01)
loss=tf.keras.losses.BinaryCrossentropy()
#compile the model
#jismein optimizer weights update karega, loss function error calculate karega,
# aur metrics training progress dikhayega.
models.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])
#set up the tensorboard
#datetime... → Har bar new folder banega with current time, taake purane logs delete na hoon.
log_dir="logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)
#set up early stopping 
#EarlyStopping ek callback hota hai jo training ko automatically rok deta hai,
#  agar model ki performance improve nahi ho rahi ho.
earlystopping_callback = EarlyStopping(monitor="val_loss",     # kya monitor karna hai
                                       patience=10,              # kitne epochs tak wait karna hai
                                       restore_best_weights=True) # best wale weights ko wapas lana hai ya nahi
#traning the model 
#10 epochs = Model ne data ko 10 baar repeat karke training ki.
#📌 Zyada epochs → Model zyada seekhta hai
#Validation Data	To tune the model (check performance after each epoch).
history=models.fit(
    x_train,y_train,validation_data=(x_test,y_test),epochs=100,
    #Callbacks wo functions hote hain jo model ki 
    # training ke dauran automatically chalte hain, earlystoping ka tensorboard ka kam krta 
    callbacks=[tensorflow_callback,earlystopping_callback]
)
models.save("models.h5")

# load tensor board extention
#%load_ext tensorboard
#tensorboard --logdir=logs/fit
#run vs code    python3 -m tensorboard.main --logdir=logs/fit20250724-155344

print(x_train.columns)  # during training



  