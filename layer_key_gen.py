#//////////////////////////////////////////////////////////////////////
#This programm generates all the possible layer keys from a input model
#//////////////////////////////////////////////////////////////////////


# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
from pprint import pprint #ADDED THIS TO PRINT ARRAY NICELY!

import sys
#You will probably have to change this!
sys.path.append("/home/veli/src/caffe/distribute/python")
import caffe

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


#ANIMAL
#PLEASE MAKE SURE TO SELECT THE RIGHT MODEL FOR THE KEYS!!!
model_path = '/home/veli/src/caffe/models/bvlc_googlenet/' # substitute your path here >> this are my settings so you have to change them
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

#bvlc_alexnet model
#Here you select the model
#model_path = '/home/veli/src/caffe/models/bvlc_alexnet/' # substitute your path here >> this are my settings so you have to change them
#net_fn   = model_path + 'deploy.prototxt'
#param_fn = model_path + 'bvlc_alexnet.caffemodel'

#PLACES model
#DIFFERENT MODEL (install first!)
#model_path = '/home/veli/src/caffe/models/googlenet_places205/' # substitute your path here >> this are my settings so you have to change them
#net_fn   = model_path + 'deploy_places205.protxt'
#param_fn = model_path + 'googlelet_places205_train_iter_2400000.caffemodel'

#PLACES 365 model
#DIFFERENT MODEL (install first!)
#model_path = '/home/veli/src/caffe/models/GoogLeNet-places365/' # substitute your path here >> this are my settings so you have to change them
#net_fn   = model_path + 'deploy_googlenet_places365.prototxt'
#param_fn = model_path + 'googlenet_places365.caffemodel'


#recent mobilenet model
#DIFFERENT MODEL (install first!)
#model_path = '/home/veli/src/caffe/models/google_mobilenet/' # substitute your path here >> this are my settings so you have to change them
#net_fn   = model_path + 'mobilenet_deploy.prototxt'
#param_fn = model_path + 'mobilenet.caffemodel'

#GoogleNet_SOS model
#model_path = '/home/veli/src/caffe/models/GoogleNet_SOS/' # substitute your path here >> this are my settings so you have to change them
#net_fn   = model_path + 'deploy.prototxt'
#param_fn = model_path + 'GoogleNet_SOS.caffemodel'

#GoogleNet-cars
#model_path = '/home/veli/src/caffe/models/GoogleNet-cars/' # substitute your path here >> this are my settings so you have to change them
#net_fn   = model_path + 'deploy.prototxt'
#param_fn = model_path + 'googlenet_finetune_web_car_iter_10000.caffemodel'

#Places_CNDS_models
#model_path = '/home/veli/src/caffe/models/Places_CNDS_models/' # substitute your path here >> this are my settings so you have to change them
#net_fn   = model_path + 'deploy.prototxt'
#param_fn = model_path + '8conv3fc_DSN.caffemodel'

#BirdSnap
#model_path = '/home/veli/src/caffe/models/BirdSnap/' # substitute your path here >> this are my settings so you have to change them
#net_fn   = model_path + 'GoogleNet_birdsnap_6.prototxt'
#param_fn = model_path + 'GoogleNet_birdsnap_6.caffemodel'


# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


#PRINT KEYS (layers)
pprint(net.blobs.keys())
print 'PLEASE, HERE IS THE LIST of all the layers (make sure to select the right model in the source!!)!!'
