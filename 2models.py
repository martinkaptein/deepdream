# Still work in progress!

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

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


#ANIMAL model (default)
#Here you select the model
model_path = '/home/veli/src/caffe/models/bvlc_googlenet/' # substitute your path here >> this are my settings so you have to change them
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'



#bvlc_alexnet model >> not working yet
#Here you select the model
#model_path = '/home/veli/src/caffe/models/bvlc_alexnet/' # substitute your path here >> this are my settings so you have to change them
#net_fn   = model_path + 'deploy.prototxt'
#param_fn = model_path + 'bvlc_alexnet.caffemodel'

#PLACES model
#model_path = '/home/veli/src/caffe/models/googlenet_places205/' # substitute your path here >> this are my settings so you have to change them
#net_fn   = model_path + 'deploy_places205.protxt'
#param_fn = model_path + 'googlelet_places205_train_iter_2400000.caffemodel'

#PLACES 365 model
#model_path = '/home/veli/src/caffe/models/GoogLeNet-places365/' # substitute your path here >> this are my settings so you have to change them
#net_fn   = model_path + 'deploy_googlenet_places365.prototxt'
#param_fn = model_path + 'googlenet_places365.caffemodel'

#recent mobilenet model >> not working yet
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



def objective_L2(dst):
    dst.diff[:] = dst.data 

#Jitter is best set to 0 in my experience >> ?
def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True, objective=objective_L2):
    
#function BAK def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)    





def deepdream(net, base_img, iter_n=11, octave_n=4, octave_scale=1.4, 
              end='inception_4c/output', clip=True, **step_params):
    #BACKUP high detail: def deepdream(net, base_img, iter_n=12, octave_n=6, octave_scale=1.6,end='inception_5b/pool_proj', clip=True, **step_params):
    #deepdream(net, base_img, iter_n=10, octave_n=7, octave_scale=1.6,end='prob', clip=False, **step_params):
    #function params>>net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_5b/5x5', clip=True, **step_params
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            showarray(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])



#///////////////////////////////////////////////////////////////
img = np.float32(PIL.Image.open('source_pictures/elbrusw1000.jpg'))
#///////////////////////////////////////////////////////////////

#folder already there, but just in case I include it (just uncomment)
#!mkdir output
frame = img
frame_i = 0


#h, w = frame.shape[:2]
#s = 0.05 # scale coefficient
for i in xrange(200):
    frame = deepdream(net, frame)
    PIL.Image.fromarray(np.uint8(frame)).save("output/%04d.jpg"%frame_i)
    #This helps get a better result:
    frame = np.float32(PIL.Image.open('output/%04d.jpg'%frame_i))
    #frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
    frame_i += 1

