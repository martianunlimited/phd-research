import foolbox
from foolbox.models import KerasModel
from foolbox.attacks import LBFGSAttack
from foolbox.criteria import TargetClassProbability
from foolbox.criteria import TopKMisclassification
from foolbox.criteria import ConfidentMisclassification
import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt

# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')

preprocessing = (np.array([103.0626,115.9029,123.1516]), 1)
fmodel = KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)


attacks=['GradientAttack',
'GradientSignAttack',
'IterativeGradientAttack',
'IterativeGradientSignAttack',
'LBFGSAttack',
'ApproximateLBFGSAttack',
'DeepFoolAttack',
'DeepFoolL2Attack',
'DeepFoolLinfinityAttack',
'SaliencyMapAttack',
'GaussianBlurAttack',
'ContrastReductionAttack',
'SinglePixelAttack',
'LocalSearchAttack',
'SLSQPAttack',
'AdditiveUniformNoiseAttack',
'AdditiveGaussianNoiseAttack',
'BlendedUniformNoiseAttack',
'SaltAndPepperNoiseAttack',
'BoundaryAttack',
'PointwiseAttack',
'BinarizationRefinementAttack',
'NewtonFoolAttack',
'ADefAttack',
'SpatialAttack',
'CarliniWagnerL2Attack',
'LinfinityBasicIterativeAttack',
'BasicIterativeMethod',
'L1BasicIterativeAttack',
'L2BasicIterativeAttack',
'ProjectedGradientDescentAttack',
'ProjectedGradientDescent',
'RandomStartProjectedGradientDescentAttack',
'RandomProjectedGradientDescent',
'MomentumIterativeAttack',
'MomentumIterativeMethod']

image=cv2.imread('myTest7.JPEG')
imageS=cv2.resize(image,dsize=(224,224)).astype(np.float)
print('predicted class', np.argmax(fmodel.predictions(imageS[:,:,:])))
label=np.argmax(fmodel.predictions(imageS[:,:,:]))

for atk in attacks:
   attack=eval('foolbox.attacks.'+atk+'(fmodel, criterion=TargetClassProbability(388, p=.8))')
   adversarial = attack(imageS[:,:,:], label)
   if type(adversarial)!=np.ndarray:
      print(atk+": Targeted Class not Supported")
      attack=eval('foolbox.attacks.'+atk+'(fmodel, criterion=TopKMisclassification(2))')
      adversarial = attack(imageS[:,:,:], label)
   if type(adversarial)!=np.ndarray:
      print(atk+": Targeted Class not Supported")
      attack=eval('foolbox.attacks.'+atk+'(fmodel, criterion=ConfidentMisclassification(0.5))')
      adversarial = attack(imageS[:,:,:], label)
   if type(adversarial)==np.ndarray:
      sum(sum(adversarial[:,:,:]-imageS))
      diff = adversarial[:,:,:]-imageS
      print(np.argmax(fmodel.predictions(adversarial)))
      advLabel=np.argmax(fmodel.predictions(adversarial))
      print(foolbox.utils.softmax(fmodel.predictions(adversarial))[label])
      print(foolbox.utils.softmax(fmodel.predictions(adversarial))[advLabel])
      adversarial=adversarial[:,:,::-1]
      adversarial_rgb = adversarial[np.newaxis, :, :, :]
      preds = kmodel.predict(preprocess_input(adversarial_rgb.copy()))
      print("Top 5 predictions (adversarial: ", decode_predictions(preds, top=5))
      preds = kmodel.predict(preprocess_input(imageS[np.newaxis,:,:,::-1].copy()))
      print("Top 5 predictions (Original: ", decode_predictions(preds, top=5))
      sio.savemat('advers'+atk+'_hen.mat',{'im':adversarial, 'diff':diff})


image=cv2.imread('myTest8.JPEG')
imageS=cv2.resize(image,dsize=(224,224)).astype(np.float)
print('predicted class', np.argmax(fmodel.predictions(imageS[:,:,:])))
label=np.argmax(fmodel.predictions(imageS[:,:,:]))

for atk in attacks:
   attack=eval('foolbox.attacks.'+atk+'(fmodel, criterion=TargetClassProbability(8, p=.8))')
   adversarial = attack(imageS[:,:,:], label)
   if type(adversarial)!=np.ndarray:
      print(atk+": Targeted Class not Supported")
      attack=eval('foolbox.attacks.'+atk+'(fmodel, criterion=TopKMisclassification(2))')
      adversarial = attack(imageS[:,:,:], label)
   if type(adversarial)!=np.ndarray:
      print(atk+": Targeted Class not Supported")
      attack=eval('foolbox.attacks.'+atk+'(fmodel, criterion=ConfidentMisclassification(0.5))')
      adversarial = attack(imageS[:,:,:], label)
   if type(adversarial)==np.ndarray:
      sum(sum(adversarial[:,:,:]-imageS))
      diff = adversarial[:,:,:]-imageS
      print(np.argmax(fmodel.predictions(adversarial)))
      advLabel=np.argmax(fmodel.predictions(adversarial))
      print(foolbox.utils.softmax(fmodel.predictions(adversarial))[label])
      print(foolbox.utils.softmax(fmodel.predictions(adversarial))[advLabel])
      adversarial=adversarial[:,:,::-1]
      adversarial_rgb = adversarial[np.newaxis, :, :, :]
      preds = kmodel.predict(preprocess_input(adversarial_rgb.copy()))
      print("Top 5 predictions (adversarial: ", decode_predictions(preds, top=5))
      preds = kmodel.predict(preprocess_input(imageS[np.newaxis,:,:,::-1].copy()))
      print("Top 5 predictions (Original: ", decode_predictions(preds, top=5))
      sio.savemat('advers'+atk+'_panda.mat',{'im':adversarial, 'diff':diff})





