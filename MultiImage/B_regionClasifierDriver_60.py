'''
This is the Classifier Driver for the detected regions
It uses
1. B_A_Classifier - Uses pre-trained parameters 6060_1000_200_0.01_10.0.pkl and regions detected test.pkl
2. B_B_Evaluator - Orders the image set based on matches with detected regions
'''

import cPickle
import numpy as np
import theano
import theano.tensor as T

import B_A_Classifier as clsfr
import B_B_Evaluator as evaluator

noMultiImages = 10


'''inpImages = ['multi1','multi2','multi3','multi4','multi5',
             'multi6','multi7','multi8','multi9','multi10']'''

inpComp = [[1,2,8],[1,2,8],[4,8,9],[2,6,7],[1,3,9],[1,8,9],[0,1,9],[0,1,9],[4,6,7],[0,2,2]]

cifar10 = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def order(predLabel):
    ''' Ordering of Composite Images '''

    inpOrder = [0]*noMultiImages
    #print predLabel
    for i in xrange(noMultiImages):
        for j in predLabel:
            if j in inpComp[i]:
                inpOrder[i]+=1
    #print inpOrder
    return inpOrder


if __name__ == '__main__' :

    # ----------------- LOADING TEST DATA ATTRIBUTES --------------
    testDataFileName = "test.pkl"          #tochange
    fileparam = "6060_1000_200_0.01_10.0.pkl"   #tochange
    mbSize = 10                                #tochange
    
    testDataFile = open(testDataFileName,"rb")
    imageList,noBB,regionProposed,testData, imageObjClass, imageObjBB, imageObjno = cPickle.load(testDataFile)

    print "No of Images :",len(imageList)
    '''print noBB
    print regionProposed
    print testData
    print imageObjClass
    print imageObjBB
    print imageObjno'''
    #raw_input("WAIT::")

    print len(noBB),len(regionProposed),len(testData)
    noRegionsTotal = len(testData)
    testData = theano.tensor._shared(np.asarray(testData,dtype=theano.config.floatX),borrow=True)
    print "----------------------testData Loaded-------------------------"

    # ----------------- LOADING CLASSIFIER PARAMS --------------
    #classifierParamFileName = "../whole_cifar10/params/"+fileparam
    classifierParamFileName = fileparam
    classifierparamFile = open(classifierParamFileName,"rb")
    params = cPickle.load(classifierparamFile)
    noOfLayers = len(params)/2 #w,b
    '''for i in params:
        print i.shape'''
    print "----------------------CLASSIFIER PARAMS Loaded-------------------------"


    # ----------------- SETTING CLASSIFIER ATTRIBUTES --------------
    noOfClasses = 10
    inpChannel = 3
    imgDimension = (32,32)
    noFilters1 = 60
    noFilters2 = 60
    fiterDimension = (5,5)
    poolDimension = (2,2)
    strideLength = 1
    #FullyConnected Layer Params
    # (I - F)/S + 1 
    cLayerOutNeuron = ((imgDimension[0]-fiterDimension[0])/strideLength+1,
                       (imgDimension[1]-fiterDimension[1])/strideLength+1
                       )
    fLayerOutNeuron = 1000
    sLayerInpNeuron = fLayerOutNeuron
    sLayerOutNeuron = noOfClasses

    cpLayer1 = clsfr.ConvPoolLayer(imgShape=(mbSize,inpChannel,imgDimension[0],imgDimension[1]),
                             lrfShape=(noFilters1,inpChannel,fiterDimension[0],fiterDimension[1]),
                             weight = params[0],
                             bias = params[1],
                             poolSize=poolDimension,
                             activation=clsfr.ReLU
                            )
    cpLayer2 = clsfr.ConvPoolLayer(imgShape=(mbSize,noFilters1,14,14),
                            lrfShape=(noFilters2,noFilters1,fiterDimension[0],fiterDimension[1]),
                             weight =params[2],
                             bias =params[3],
                             poolSize=poolDimension,
                             activation=clsfr.ReLU
                            )
    fLayer = clsfr.FullyConnectedLayer(nInp=noFilters2*5*5,
                                       nOut=fLayerOutNeuron,
                                         weight =params[4],
                                         bias = params[5],
                                         activation=clsfr.ReLU)
    smLayer = clsfr.SoftmaxLayer(nInp=sLayerInpNeuron,
                           nOut=sLayerOutNeuron,
                             weight =params[6],
                             bias = params[7])

    netArch = [cpLayer1, cpLayer2, fLayer, smLayer]
    n = clsfr.Network(netArch, mbSize)
    
    

    #----------------- EXECUTING CLASSIFIER ---------------
    
    predOP,predScoreMatrix = n.mbSGDTest([testData], mbSize)

    print predOP
    #print predScoreMatrix 
    print "Pred[0] shape",predScoreMatrix[0].shape
    print "Len of regionProposed",len(regionProposed)
    print "Len of noBB",len(noBB)
    print "Len of predScoreMatrix",len(predScoreMatrix)
    #raw_input("B_ WAIT:")
    
    
    regionProposedArr = np.zeros((noRegionsTotal,4))
    predScoreMatrixArr = np.zeros((noRegionsTotal,10))
    print "Predicted Array zeros shape :",regionProposedArr.shape

    #all images 300
    startBB = 0
    endBB = 0
    for (imgg,no) in zip(regionProposed,noBB):
        endBB=endBB+no
        #print startBB,endBB
        '''print regionProposedArr[startBB:endBB,:]
        print regionProposedArr[startBB:endBB,:].shape
        print np.array(imgg)
        print np.array(imgg).shape'''
        if no == 0 :
            continue
        regionProposedArr[startBB:endBB,:]=np.array(imgg)
        startBB=endBB
    print regionProposedArr.shape


    # all regions 13475
    startBB = 0
    endBB = 0
    for j in predScoreMatrix:
        #print j.shape
        endBB=endBB+len(j)
        predScoreMatrixArr[startBB:endBB,:]=j
        startBB =endBB
    print predScoreMatrixArr.shape
        

    
    #raw_input("WAIT:: 6666666")


    #----------------- EVALUATION ---------------------

    noTopClass = 3
    noTopScoredRegionPerImg = 5
    overlapRatio = 0.5

    matched,total,matchPercent,predOP = evaluator.accuracy(imageList, noBB, regionProposedArr, predScoreMatrixArr,imageObjBB,imageObjClass,noTopClass,noTopScoredRegionPerImg,overlapRatio)

    print "noTopScoredRegionPerImg : ",noTopScoredRegionPerImg
    print 'Matched : ',matched
    print 'Total : ',total
    print 'Accuracy : ',matchPercent,"%"
    print predOP

    '''for i in xrange(noTopScoredRegionPerImg,105,5):
        matched,total,matchPercent = evaluator.accuracy(imageList, noBB, regionProposedArr, predScoreMatrixArr,imageObjBB,imageObjClass,noTopClass,i,overlapRatio)

        print "noTopScoredRegionPerImg : ",i
        print 'Matched : ',matched
        print 'Total : ',total
        print 'Accuracy : ',matchPercent,"%"'''

    '''print imageList
    print imageObjBB
    print imageObjClass
    print regionProposedArr
    print predScoreMatrixArr
    print noBB'''

    #----------------- ORDERING ---------------------

    for x in xrange(10):
        testImageIdx = x
        #testImage = inpImages[int(testImageIdx)]
        orderList = order(predOP[x])
        print 'Conv Neural Network order : ',orderList
        zz =[-i+1 for (v,i) in sorted(((v, -i) for (i, v) in enumerate(orderList) if v>0 and i!=int(testImageIdx)),reverse=True)]
        print "zz",zz
        print [cifar10[i] for i in predOP[x]]
        print [orderList[int(i-1)] for i in zz]
        print "-----------------------------------------"
    

    
