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
from keras.callbacks import TensorBoard, EarlyStopping
import tensorflow as tf

nb_classes=1000
nb_obs=50000
batch_size = 100
ensMax=1
subspaces=[100000]
splits=0
shuffle=True
dirCount=1
runCount=1
ensSizes=np.arange(ensMax)+1
alexDir=np.arange(24)+1;
googleDir=np.arange(26)+1;
res50Dir=np.arange(18)+1;
res152Dir=np.arange(15)+1;


num_batches=np.floor(nb_obs/batch_size)
#indices=np.arange(nb_obs)
#np.random.shuffle(indices)
#trainSize=int(splits*nb_obs)
#[trainIdx,testIdx]=[indices[:trainSize],indices[trainSize:]]
data=np.load('nnIndices.npz')
trainIdx=data['trainIdx']
testIdx=data['testIdx']
data.close()


top3_acc = partial(top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'

top5_acc = partial(top_k_categorical_accuracy, k=5)
top5_acc.__name__ = 'top5_acc'

model=load_model('exp8_ens_3_subs_350.h5',custom_objects={"top3_acc":top3_acc,"top5_acc":top5_acc});
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy',top3_acc,top5_acc])

meanL=[]
stdL=[]
minL=[]
maxL=[]
medL=[]
subspaceL=[]
ensL=[]
for ensSize in ensSizes:
   ensL.append(ensSize)
   meanM=[]
   stdM=[]
   minM=[]
   maxM=[]
   medM=[]
   subspaceM=[]
   for subspace in subspaces:
      subspaceM.append(subspace)
      scores=[]
      for runNo in range(runCount):
         X1trainList=[];
         X1testList=[];
         X2trainList=[];
         X2testList=[];
         X3trainList=[];
         X3testList=[];
         X4trainList=[];
         X4testList=[];

         np.random.shuffle(alexDir)
         np.random.shuffle(googleDir)
         np.random.shuffle(res50Dir)
         np.random.shuffle(res152Dir)
         
         for i in range(ensSize):
            dir=alexDir[i]
            nnType='alexnet'
            matFile='../myMatlab/matconvnet-1.0-beta24/'+nnType+'/base/sparse.mat'
            M=scio.loadmat(matFile)
            X=M["best"]
            X.astype(float)
            X1trainList.append(X[trainIdx,:])
            X1testList.append(X[testIdx,:])
            dir=googleDir[i]
            nnType='googlenet'
            matFile='../myMatlab/matconvnet-1.0-beta24/'+nnType+'/base/sparse.mat'
            M=scio.loadmat(matFile)
            X=M["best"]
            X.astype(float)
            X2trainList.append(X[trainIdx,:])
            X2testList.append(X[testIdx,:])
            nnType='resnet50'
            dir=res50Dir[i]
            matFile='../myMatlab/matconvnet-1.0-beta24/'+nnType+'/base/sparse.mat'
            M=scio.loadmat(matFile)
            X=M["best"]
            X.astype(float)
            X3trainList.append(X[trainIdx,:])
            X3testList.append(X[testIdx,:])
            nnType='resnet152'
            dir=res152Dir[i]
            matFile='../myMatlab/matconvnet-1.0-beta24/'+nnType+'/base/sparse.mat'
            M=scio.loadmat(matFile)
            X=M["best"]
            X.astype(float)
            X4trainList.append(X[trainIdx,:])
            X4testList.append(X[testIdx,:])
         
         M=scio.loadmat('solnArry.mat')
         Y=M["solnArry"]
         Y_train=Y[trainIdx,:]
         Y_test=Y[testIdx,:]
         
         sumX1val=X1testList[0].toarray()
         sumX2val=X2testList[0].toarray()
         sumX3val=X3testList[0].toarray()
         sumX4val=X4testList[0].toarray()
         maxX1val=X1testList[0].toarray()
         maxX2val=X2testList[0].toarray()
         maxX3val=X3testList[0].toarray()
         maxX4val=X4testList[0].toarray()
         for i in range(1,len(X1trainList)):
             maxX4val=np.maximum(maxX4val,X4testList[i].toarray())
             maxX3val=np.maximum(maxX3val,X3testList[i].toarray())
             maxX2val=np.maximum(maxX2val,X2testList[i].toarray())
             maxX1val=np.maximum(maxX1val,X1testList[i].toarray())
             sumX1val=sumX1val+X1testList[i].toarray()
             sumX2val=sumX2val+X2testList[i].toarray()
             sumX3val=sumX3val+X3testList[i].toarray()
             sumX4val=sumX4val+X4testList[i].toarray()
         Y_val=np_utils.to_categorical(Y_test-1,1000)
         
         score=model.evaluate([sumX1val/ensSize, sumX2val/ensSize, sumX3val/ensSize, sumX4val/ensSize,maxX1val,maxX2val,maxX3val,maxX4val],Y_val, verbose=1)
         scores.append(score) 
      stdM.append(np.std(scores,axis=0))
      meanM.append(np.mean(scores,axis=0))
      medM.append(np.median(scores,axis=0))
      minM.append(np.min(scores,axis=0))
      maxM.append(np.max(scores,axis=0))
   stdL.append(stdM)   
   meanL.append(meanM)   
   medL.append(medM)   
   minL.append(minM)   
   maxL.append(maxM)   
   subspaceL.append(subspaceM)

meanL=np.array(meanL)
stdL=np.array(stdL)
minL=np.array(minL)
maxL=np.array(maxL)
medL=np.array(medL)
ensL=np.array(ensL)
subspaceL=np.array(subspaceL)

print(meanL)
print(stdL)
print(medL)
print(minL)
print(maxL)
np.savez('nnBase.npz',meanL=meanL,stdL=stdL,medL=medL,minL=minL,maxL=maxL,ensL=ensL,subspaceL=subspaceL)
scio.savemat('nnBase.mat',{'meanL':meanL,'stdL':stdL,'medL':medL,'minL':minL,'maxL':maxL,'ensL':ensL,'subspaceL':subspaceL})
