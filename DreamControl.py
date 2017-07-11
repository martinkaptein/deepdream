

#SCROLL DOWN FOR SETTINGS (Google's Deep Dream Notebook code modified by me (@MartinKaptein) and converted to py script for convenience)


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



#Here you select the model
model_path = '/home/veli/src/caffe/models/bvlc_googlenet/' # substitute your path here >> this are my settings so you have to change them
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

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

#Jitter is best set to 0 in my experience
#SOME LITTLE SETTINGS
def make_step(net, step_size=1.6, end='inception_5b/output', jitter=0, clip=True, objective=objective_L2):
    
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




#///////////////////////////////////////////////////////////////////////////
#MAIN DEEPDREAM SETTINGS
#///////////////////////////////////////////////////////////////////////////
def deepdream(net, base_img, iter_n=12, octave_n=6, octave_scale=1.6, end='inception_5b/output', clip=True, **step_params):
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




#SELECT HERE THE PICTURE YOU WANT TO DRAW THE DREAM ON:
img = np.float32(PIL.Image.open('source_pictures/PhotoScaled.jpg'))

#SELECT HERE THE DREAM GUIDE IMAGE (RECOMMEND 224x224)
guide = np.float32(PIL.Image.open('dream_guide/drawing_girl.jpg'))

#ADVANCED DREAM CONTROL AND FUNCTION DEF
#original>>end = 'inception_3b/output'
end = 'inception_5b/output'
h, w = guide.shape[:2]
src, dst = net.blobs['data'], net.blobs[end]
src.reshape(1,3,h,w)
src.data[0] = preprocess(net, guide)
net.forward(end=end)
guide_features = dst.data[0].copy()


def objective_guide(dst):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

#deepdream(net, img, end=end, objective=objective_guide)

#GO
frame = img
counter = 0
for i in xrange(8): #change how many times you want to re-feed
    frame = deepdream(net, frame, end=end, objective=objective_guide) #HERE HAPPENS THE MAJORITY OF THE WORK :)
    PIL.Image.fromarray(np.uint8(frame)).save("output/dc%04d.jpg"%counter)
    #Open the file we just created (again)
    frame = np.float32(PIL.Image.open('output/dc%04d.jpg'%counter))
    counter += 1


print 'Done!'



