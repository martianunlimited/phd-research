from scipy import io as scio
from scipy import sparse as sparse
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

nb_classes=1000
nb_obs=50000
batch_size = 200
nnType='googlenet2'
ensSize=3
subspace=350
splits=0.8
shuffle=True

num_batches=np.floor(nb_obs/batch_size)
indices=np.arange(nb_obs)
np.random.shuffle(indices)
trainSize=int(splits*nb_obs)
[trainIdx,testIdx]=[indices[:trainSize],indices[trainSize:]]
X1trainList=[];
X1testList=[];
X2trainList=[];
X2testList=[];
X3trainList=[];
X3testList=[];
X4trainList=[];
X4testList=[];

for i in range(ensSize):
   dir=i+1;
   nnType='alexnet'
   matFile='../myMatlab/matconvnet-1.0-beta24/'+nnType+'/run_'+str(dir)+'/sparse_'+str(subspace)+'.mat'
   M=scio.loadmat(matFile)
   X=M["best"]
   X.astype(float)
   X1trainList.append(X[trainIdx,:])
   X1testList.append(X[testIdx,:])
   nnType='googlenet'
   matFile='../myMatlab/matconvnet-1.0-beta24/'+nnType+'/run_'+str(dir)+'/sparse_'+str(subspace)+'.mat'
   M=scio.loadmat(matFile)
   X=M["best"]
   X.astype(float)
   X2trainList.append(X[trainIdx,:])
   X2testList.append(X[testIdx,:])
   nnType='resnet50'
   matFile='../myMatlab/matconvnet-1.0-beta24/'+nnType+'/run_'+str(dir)+'/sparse_'+str(subspace)+'.mat'
   M=scio.loadmat(matFile)
   X=M["best"]
   X.astype(float)
   X3trainList.append(X[trainIdx,:])
   X3testList.append(X[testIdx,:])
   nnType='resnet152'
   matFile='../myMatlab/matconvnet-1.0-beta24/'+nnType+'/run_'+str(dir)+'/sparse_'+str(subspace)+'.mat'
   M=scio.loadmat(matFile)
   X=M["best"]
   X.astype(float)
   X4trainList.append(X[trainIdx,:])
   X4testList.append(X[testIdx,:])

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
       maxX1=X1trainList[0][batch_index,:].toarray()
       sumX1=X1trainList[0][batch_index,:].toarray()
       maxX2=X2trainList[0][batch_index,:].toarray()
       sumX2=X2trainList[0][batch_index,:].toarray()
       maxX3=X3trainList[0][batch_index,:].toarray()
       sumX3=X3trainList[0][batch_index,:].toarray()
       maxX4=X4trainList[0][batch_index,:].toarray()
       sumX4=X4trainList[0][batch_index,:].toarray()
       for i in range(1,len(X1trainList)):
          maxX4=np.maximum(maxX4,X4trainList[i][batch_index,:].toarray())
          maxX3=np.maximum(maxX3,X3trainList[i][batch_index,:].toarray())
          maxX2=np.maximum(maxX2,X2trainList[i][batch_index,:].toarray())
          maxX1=np.maximum(maxX1,X1trainList[i][batch_index,:].toarray())
          sumX4=sumX4+X4trainList[i][batch_index,:].toarray()
          sumX3=sumX3+X3trainList[i][batch_index,:].toarray()
          sumX2=sumX2+X2trainList[i][batch_index,:].toarray()
          sumX1=sumX1+X1trainList[i][batch_index,:].toarray()
       Y_batch=Y_train[batch_index]-1
       yield [sumX1/ensSize, sumX2/ensSize, sumX3/ensSize, sumX4/ensSize, maxX1,maxX2,maxX3,maxX4] , Y_batch
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
       batch_index=smp_idx
       maxX1val=X1testList[0][batch_index,:].toarray()
       sumX1val=X1testList[0][batch_index,:].toarray()
       maxX2val=X2testList[0][batch_index,:].toarray()
       sumX2val=X2testList[0][batch_index,:].toarray()
       maxX3val=X3testList[0][batch_index,:].toarray()
       sumX3val=X3testList[0][batch_index,:].toarray()
       maxX4val=X4testList[0][batch_index,:].toarray()
       sumX4val=X4testList[0][batch_index,:].toarray()
       for i in range(len(X1testList)):
          maxX4val=np.maximum(maxX4val,X4testList[i][batch_index,:].toarray())
          maxX3val=np.maximum(maxX3val,X3testList[i][batch_index,:].toarray())
          maxX2val=np.maximum(maxX2val,X2testList[i][batch_index,:].toarray())
          maxX1val=np.maximum(maxX1val,X1testList[i][batch_index,:].toarray())
          sumX4val=sumX4val+X4testList[i][batch_index,:].toarray()
          sumX3val=sumX3val+X3testList[i][batch_index,:].toarray()
          sumX2val=sumX2val+X2testList[i][batch_index,:].toarray()
          sumX1val=sumX1val+X1testList[i][batch_index,:].toarray()
       Y_batchVal=Y_test[batch_index]-1
       yield [sumX1val/ensSize,sumX2val/ensSize,sumX3val/ensSize,sumX4val/ensSize,maxX1val,maxX2val,maxX3val,maxX4val] , Y_batchVal
       if shuffle:
          np.random.shuffle(smp_idx)

G=myGeneratorTrain()
G1=myGeneratorVal()
clf=OneVsRestClassifier(SGDClassifier(loss='log',penalty='l1',max_iter=1000,n_jobs=-1),n_jobs=-2)
for i in range(5):
    print("i=",i)
    for j in range(int(num_batches*splits)):
        K=G.next()
        yt=K[1]
        xt=np.concatenate((K[0][0],K[0][1],K[0][2],K[0][3],K[0][4],K[0][5],K[0][6],K[0][7]),axis=1)        
        clf.partial_fit(xt,yt,classes=np.unique(Y)-1)
    K1=G1.next()
    yv=K1[1]
    xv=np.concatenate((K1[0][0],K1[0][1],K1[0][2],K1[0][3],K1[0][4],K1[0][5],K1[0][6],K1[0][7]),axis=1)        
    print(clf.score(xv,yv))

from sklearn.externals import joblib
joblib.dump(clf,'sgdClassifier.pkl')

np.savez('nnIndices2.npz',trainIdx=trainIdx,testIdx=testIdx);

#model.evaluate([X_test[:,0:1000],X_test[:,1000:2000],X_test[:,2000:3000]],np_utils.to_categorical(Y[testIdx]-1,1000))

