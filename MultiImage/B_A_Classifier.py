
import numpy as np
import gzip
import cPickle


import theano
import theano.tensor as T

from theano.tensor.nnet import conv
from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import softmax

#from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor import shared_randomstreams

import time




def ReLU(z):
    return T.maximum(0.0, z)

def dropoutLayer(layer, pDropout):
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-pDropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)

def size(data):
    return data[0].get_value(borrow=True).shape[0]


class ConvPoolLayer(object):

    def __init__(self, lrfShape, imgShape, weight, bias, poolSize=(2,2), activation=sigmoid):
        self.lrfShape = lrfShape
        self.imgShape = imgShape
        self.poolSize = poolSize
        self.activation = activation
        #print "Hi"
        print 'Filter Shape : ',self.lrfShape
        print 'Image Shape : ',self.imgShape
        print 'Maxpool Size : ',self.poolSize
        print 'Activation : ',self.activation

        noOutNeuron = ( lrfShape[0]*np.prod(lrfShape[2:])/np.prod(poolSize) )
        #print "Neurons for weights of filter : ",noOutNeuron   #4*(5*5*3)/(2*2)
        
        self.w = theano.shared(weight, borrow=True)
        self.b = theano.shared(bias, borrow=True)
        self.params = [self.w, self.b]


    def setInputOutput(self, inp, inpDropout, mbSize):
        
        self.inp = inp.reshape(self.imgShape)
        convOutput = conv.conv2d( input=self.inp,
                                       filters=self.w,
                                       filter_shape=self.lrfShape,
                                       image_shape=self.imgShape
                                  )
        #ds - downsize
        pooledOutput = pool.pool_2d( input=convOutput,
                                               ds=self.poolSize, 
                                               ignore_border=True
                                               )
        self.out = self.activation( pooledOutput + self.b.dimshuffle('x',0,'x','x') )
        self.outDropout = self.out     #nodropout convlayer
    

class FullyConnectedLayer(object):

    def __init__(self, nInp, nOut, weight, bias, activation=sigmoid, pDropout=0.0):
        self.nInp = nInp
        self.nOut = nOut
        self.activation = activation
        self.pDropout = pDropout

        self.w = theano.shared( weight,
                                name = 'w',
                                borrow = True
                                )
        self.b = theano.shared( bias,
                                name = 'b',
                                borrow = True
                                )
        self.params = [self.w, self.b]


    def setInputOutput(self, inp, inpDropout, mbSize):
        self.inp = inp.reshape(( mbSize, self.nInp ))
        self.out = self.activation( (1.0-self.pDropout)*T.dot(self.inp,self.w)+self.b )
        self.yOut = T.argmax( self.out,axis=1 )

        self.inpDropout = dropoutLayer( inpDropout.reshape(( mbSize, self.nInp )),
                                        self.pDropout )
        self.outDropout = self.activation( T.dot(self.inpDropout,self.w)+self.b )

        
    def accuracy(self, y):
        return  T.mean(T.eq(self.yOut,y))


class SoftmaxLayer(object):

    def __init__(self, nInp, nOut, weight, bias, pDropout=0.0):
        self.nInp = nInp
        self.nOut = nOut
        self.pDropout = pDropout
        self.w = theano.shared(weight,
                               name = 'w',
                               borrow = True
                               )
        self.b = theano.shared(bias,
                               name = 'b',
                               borrow = True
                               )
        self.params = [self.w, self.b]


    def setInputOutput(self, inp, inpDropout, mbSize):
        self.inp = inp.reshape((mbSize, self.nInp))
        self.out = softmax( (1-self.pDropout)*T.dot(self.inp,self.w) + self.b )
        self.yOut = T.argmax(self.out, axis=1)

        self.inpDropout = dropoutLayer( inpDropout.reshape((mbSize, self.nInp)),
                                         self.pDropout )
        self.outDropout = softmax( T.dot(self.inpDropout,self.w)+self.b)


    def cost(self, net):
        ''' Log-likelihood cost'''
        return -T.mean( T.log(self.outDropout)[T.arange(net.y.shape[0]), net.y] )

    
    def accuracy(self, y):
        return  T.mean(T.eq(self.yOut,y))



class Network(object):

    def __init__(self, layers, mbSize):

        self.layers = layers
        self.mbSize = mbSize
        self.params = [ p for layer in self.layers for p in layer.params]
        print "Layers : ",self.layers
        print "MiniBatch Size : ",self.mbSize
        print "Params : ",self.params
        
        #self.x = T.matrix("x")
        self.x = T.ftensor3("x")
        self.y = T.ivector("y")

        initLayer = self.layers[0] #ConvPoolLayer
        initLayer.setInputOutput(self.x, self.x, self.mbSize)
        
        for j in xrange(1, len(self.layers)):
            prevLayer, currLayer = self.layers[j-1], self.layers[j]
            currLayer.setInputOutput(prevLayer.out, prevLayer.outDropout, self.mbSize)
        self.out = self.layers[-1].out
        self.outDropout = self.layers[-1].outDropout


    def mbSGDTest(self, testData, mbSize ):

        testX = testData[0]
        print testX
        noTestBatch = size(testData)/mbSize
        print "Test Batch = ",noTestBatch

        i = T.lscalar() #mb index
        
        '''testMBAccuracy = theano.function( [i],
                                           self.layers[-1].accuracy(self.y),
                                           givens = { self.x: testX[i*self.mbSize: (i+1)*self.mbSize],
                                                      self.y: testY[i*self.mbSize: (i+1)*self.mbSize]
                                                    }
                                         )'''
        testMBPredictions = theano.function( [i],
                                             self.layers[-1].yOut,
                                             givens = { self.x: testX[i*self.mbSize: (i+1)*self.mbSize]
                                                      }
                                           )
        testscore = theano.function( [i],
                                             self.layers[-1].out,
                                             givens = { self.x: testX[i*self.mbSize: (i+1)*self.mbSize]
                                                      }
                                           )
        
        if testData :
            #testAccuracy = np.mean( [testMBAccuracy(j) for j in xrange(noTestBatch)] )
            #print("\tTest Accuracy : {0:.2%}".format(testAccuracy))
            predOP = [testMBPredictions(j) for j in xrange(noTestBatch)]
            #print predOP[len(predOP)-1]
            predscore = [testscore(j) for j in xrange(noTestBatch)]
            #print predscore[len(predscore)-1]
            '''actOP = [testActualOP(j) for j in xrange(noTestBatch)]
            print actOP'''
            #print testY[-50:].get_value()
            #print testY
            #print testY[-50:].eval()
            print "LEN predOP",len(predOP)
            print "LEN predScore",len(predscore)
            #raw_input("WAIIT 4444")

        return predOP,predscore
