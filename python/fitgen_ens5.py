from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.engine.input_layer import Input
from keras.layers.merge import Average, Maximum, Concatenate
from keras.utils import np_utils
from keras import Model
from keras import regularizers
from scipy import io as scio
from scipy import sparse as sparse
from keras import backend as K
import numpy as np

nb_classes=1000
nb_obs=50000
batch_size = 100
shuffle=True
num_batches=np.floor(nb_obs/batch_size)
indices=np.arange(nb_obs)
np.random.shuffle(indices)
splits=0.8
trainSize=int(splits*nb_obs)
[trainIdx,testIdx]=[indices[:trainSize],indices[trainSize:]]
M=scio.loadmat('best3_train_150.mat')
X=M["best3"]
M=scio.loadmat('solnArry.mat')
Y=M["solnArry"]
X.astype('float32')
X_train=X[trainIdx,:]
X_test=X[testIdx,:]
Y_train=Y[trainIdx,:]
Y_test=Y[testIdx,:]

def myGeneratorTrain():
    counter=0
    smp_idx=np.arange(trainSize)
    if shuffle:
    	np.random.shuffle(smp_idx)
    while True:
       batch_index=smp_idx[batch_size*counter:batch_size*(counter+1)]
       X_batch=X_train[batch_index,:].toarray()
       Y_batch=np_utils.to_categorical(Y_train[batch_index]-1,1000)
       yield [X_batch[:,0:1000],X_batch[:,1000:2000],X_batch[:,2000:3000]] , Y_batch
       counter+=1
       if counter == int(num_batches*splits):
          counter=0
          if shuffle:
              np.random.shuffle(smp_idx)


def myGeneratorVal():
    M=scio.loadmat('best3_val_150.mat')
    X_val=M["best3"]
    X_val=X[testIdx,:]
    M=scio.loadmat('solnArry.mat')
    Y_val=M["solnArry"]
    Y_val=Y_val[testIdx]
    X_val.astype('float32')
    counter=0
    smp_idx=np.arange(int(nb_obs*(1-splits)))
    if shuffle:
    	np.random.shuffle(smp_idx)
    while True:
       batch_index=smp_idx[batch_size*counter:batch_size*(counter+1)]
       X_batch=X_val[batch_index,:].toarray()
       Y_batch=np_utils.to_categorical(Y_val[batch_index]-1,1000)
       yield [X_batch[:,0:1000],X_batch[:,1000:2000],X_batch[:,2000:3000]] , Y_batch
       counter+=1
       if counter == int((1-splits)*num_batches):
          counter=0
          if shuffle:
              np.random.shuffle(smp_idx)

nb_hidden = 1000
input1=Input(shape=(1000,))
input2=Input(shape=(1000,))
input3=Input(shape=(1000,))

avgL=Average()([input1,input2,input3])
maxL=Maximum()([input1,input2,input3])
concatL=Concatenate()([avgL,maxL])
layer1=Dense(nb_hidden,bias_initializer='zeros', activation='relu')(concatL)
layer1=Dropout(0.75)(layer1)
out=Dense(nb_classes,bias_initializer='zeros', kernel_initializer='identity',activation='softmax')(layer1)
model=Model([input1,input2,input3],out)
#model=load_model('genFit_ens450.h5');

model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
G=myGeneratorTrain()
K=next(G)
X_i=K[0]
Y_i=K[1]
model.fit(X_i,Y_i)
debugModel=Model(inputs=model.input, outputs=model.layers[0].output)
history=model.fit_generator(myGeneratorTrain(), steps_per_epoch = 400 , epochs = 200, verbose=2, validation_data=myGeneratorVal(), validation_steps=100 )

model.evaluate([X_test[:,0:1000],X_test[:,1000:2000],X_test[:,2000:3000]],np_utils.to_categorical(Y[testIdx]-1,1000))

