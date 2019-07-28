from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.engine.input_layer import Input
from keras.layers.merge import Average, Maximum, Concatenate, Minimum 
from keras.utils import np_utils
from keras.activations import relu
from keras import Model
from keras import regularizers
from scipy import io as scio
from scipy import sparse as sparse
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from functools import partial
from keras.callbacks import TensorBoard
import tensorflow as tf

nb_classes=1000
nb_obs=50000
batch_size = 100
nnType='googlenet2'
ensSize=5
subspace=450
splits=0.8
shuffle=True

num_batches=np.floor(nb_obs/batch_size)
indices=np.arange(nb_obs)
np.random.shuffle(indices)
trainSize=int(splits*nb_obs)
[trainIdx,testIdx]=[indices[:trainSize],indices[trainSize:]]
XtrainList=[];
XtestList=[];

top3_acc = partial(top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'

top5_acc = partial(top_k_categorical_accuracy, k=5)
top5_acc.__name__ = 'top5_acc'



for i in range(ensSize):
   dir=i+1;
   matFile='../myMatlab/matconvnet-1.0-beta24/'+nnType+'/run_'+str(dir)+'/sparse_'+str(subspace)+'.mat'
   M=scio.loadmat(matFile)
   X=M["best"]
   X.astype(float)
   XtrainList.append(X[trainIdx,:])
   XtestList.append(X[testIdx,:])

M=scio.loadmat('solnArry.mat')
Y=M["solnArry"]
Y_train=Y[trainIdx,:]
Y_test=Y[testIdx,:]

def myGeneratorTrain():
    counter=0
    smp_idx=np.arange(trainSize)
    if shuffle:
    	np.random.shuffle(smp_idx)
    while True:
       batch_index=smp_idx[batch_size*counter:batch_size*(counter+1)]
       maxX=XtrainList[0][batch_index,:].toarray()
       minX=XtrainList[0][batch_index,:].toarray()
       sumX=XtrainList[0][batch_index,:].toarray()
       for i in range(1,len(XtrainList)):
          maxX=np.maximum(maxX,XtrainList[i][batch_index,:].toarray())
          minX=np.minimum(minX,XtrainList[i][batch_index,:].toarray())
          sumX=sumX+XtrainList[i][batch_index,:].toarray()
       Y_batch=np_utils.to_categorical(Y_train[batch_index]-1,1000)
       yield [sumX/ensSize, maxX, minX] , Y_batch
       counter+=1
       if counter == int(num_batches*splits):
          counter=0
          if shuffle:
              np.random.shuffle(smp_idx)


def myGeneratorVal():
    counter=0
    smp_idx=np.arange(int(nb_obs*(1-splits)))
    if shuffle:
    	np.random.shuffle(smp_idx)
    while True:
       batch_index=smp_idx[batch_size*counter:batch_size*(counter+1)]
       maxX=XtestList[0][batch_index,:].toarray()
       minX=XtestList[0][batch_index,:].toarray()
       sumX=XtestList[0][batch_index,:].toarray()
       for i in range(1,len(XtrainList)):
          maxX=np.maximum(maxX,XtestList[i][batch_index,:].toarray())
          minX=np.minimum(minX,XtestList[i][batch_index,:].toarray())
          sumX=sumX+XtestList[i][batch_index,:].toarray()
       Y_batch=np_utils.to_categorical(Y_test[batch_index]-1,1000)
       yield [sumX/ensSize, maxX, minX] , Y_batch
       counter+=1
       if counter == int((1-splits)*num_batches):
          counter=0
          if shuffle:
              np.random.shuffle(smp_idx)


def nick_crossentropy(target, output, from_logits=False, axis=-1):
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * ((tf.sqrt(output)-1) / tf.sqrt(output)), axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)

maxXval=XtestList[0].toarray()
minXval=XtestList[0].toarray()
sumXval=XtestList[0].toarray()
for i in range(1,len(XtrainList)):
    maxXval=np.maximum(maxXval,XtestList[i].toarray())
    minXval=np.minimum(minXval,XtestList[i].toarray())
    sumXval=sumXval+XtestList[i].toarray()
Y_val=np_utils.to_categorical(Y_test-1,1000)

tensorboard=TensorBoard(log_dir='./exp5', histogram_freq=1, write_graph=True, write_images=False)

nb_hidden = 1000
input1=Input(shape=(1000,))
input2=Input(shape=(1000,))
input3=Input(shape=(1000,))
concatL=Concatenate()([input1,input2,input3])
layer1=Dense(nb_hidden,bias_initializer='zeros', kernel_regularizer=regularizers.l1(0.00002) )(concatL)
layer1=LeakyReLU(alpha=0.01)(layer1)
out=Dense(nb_classes,bias_initializer='zeros',kernel_regularizer=regularizers.l1(0.00002),kernel_initializer='identity' ,activation='softmax')(layer1)
model=Model([input1,input2,input3],out)
#model=load_model('genFit_ens450.h5');

model.compile(loss=nick_crossentropy, optimizer='adadelta',metrics=['accuracy',top3_acc,top5_acc])
G=myGeneratorTrain()
K=next(G)
X_i=K[0]
Y_i=K[1]
#model.fit(X_i,Y_i)
debugModel=Model(inputs=model.input, outputs=model.layers[0].output)
#history=model.fit_generator(myGeneratorTrain(), steps_per_epoch = 400 , epochs = 100, verbose=2, validation_data=myGeneratorVal(), validation_steps=100 )
history=model.fit_generator(myGeneratorTrain(), steps_per_epoch = 400 , epochs = 100, verbose=2, validation_data=([sumXval/ensSize, maxXval, minXval],Y_val),callbacks=[tensorboard] )


#model.evaluate([X_test[:,0:1000],X_test[:,1000:2000],X_test[:,2000:3000]],np_utils.to_categorical(Y[testIdx]-1,1000))

