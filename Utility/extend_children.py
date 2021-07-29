from matplotlib.pyplot import imread
from skimage.transform import resize
#from scipy.misc import imread,imresize
from os import listdir
from os.path import splitext
from random import seed,shuffle
from time import time
from numpy import zeros
import tensorflow as tf
from tensorflow import Variable,constant,nn
weights = lambda shape: Variable(tf.random.truncated_normal(shape, stddev=0.1))
biases = lambda shape: Variable(constant(0.1, shape=shape))
conv2d = lambda x, W: nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
max_pool = lambda x: nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



def extend_children(path,ftype=False):
    allpaths = [path+'/'+child for child in listdir(path)]
    if ftype != False:
        ret = []
        for v in allpaths:
            if splitext(v)[1] == ftype:
                ret.append(v)
    else:
        ret=allpaths
    return ret