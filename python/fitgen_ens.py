from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.engine.input_layer import Input
from keras.layers.merge import Average
from keras.utils import np_utils
from keras import Model
from scipy import io as scio
from scipy import sparse as sparse
from keras import backend as K
import numpy as np

M=scio.loadmat('best3.mat')
X=M["best3"]
M=scio.loadmat('solnArry.mat')
Y=M["solnArry"]
X.astype('float32')
nb_classes=1000
nb_obs=X.shape[0]
batch_size = 100
shuffle=False
num_batches=np.floor(nb_obs/batch_size)
def myGenerator():
    counter=0
    smp_idx=np.arange(nb_obs)
    if shuffle:
    	np.random.shuffle(smp_idx)
    while True:
       batch_index=smp_idx[batch_size*counter:batch_size*(counter+1)]
       X_batch=X[batch_index,:].toarray()
       Y_batch=np_utils.to_categorical(Y[batch_size*counter:batch_size*(counter+1)]-1,1000)
       yield [X_batch[:,0:1000],X_batch[:,1000:2000],X_batch[:,2000:3000]] , Y_batch
       counter+=1
       if counter%50==0:
          print("i="+str(counter))
       if counter == num_batches:
          counter=0
          if shuffle:
              np.random.shuffle(smp_idx)


nb_hidden = 1000
input1=Input(shape=(1000,))
input2=Input(shape=(1000,))
input3=Input(shape=(1000,))
avg=Average()([input1,input2,input3])
layer1=Dense(nb_hidden,bias_initializer='zeros', kernel_initializer='identity', activation='relu')(avg)
out=Dense(nb_classes,bias_initializer='zeros', kernel_initializer='identity',activation='softmax')(layer1)
model=Model([input1,input2,input3],out)

model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
G=myGenerator()
K=next(G)
X_i=K[0]
Y_i=K[1]
model.fit(X_i,Y_i)
debugModel=Model(inputs=model.input, outputs=model.layers[0].output)
model.fit_generator(myGenerator(), steps_per_epoch = 100 , epochs = 5000, verbose=2 )

