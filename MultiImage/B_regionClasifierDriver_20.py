import cPickle
import numpy as np
import theano
import theano.tensor as T

import B_A_Classifier as clsfr
import B_B_Evaluator as evaluator



if __name__ == '__main__' :

    # ----------------- LOADING TEST DATA ATTRIBUTES --------------
    testDataFileName = "test.pkl"          #tochange
    fileparam = "2020_500_200_0.01_10.0.pkl"   #tochange
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
    classifierParamFileName = "../whole_cifar10/params/"+fileparam
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
    noFilters1 = 20
    noFilters2 = 20
    fiterDimension = (5,5)
    poolDimension = (2,2)
    strideLength = 1
    #FullyConnected Layer Params
    # (I - F)/S + 1 
    cLayerOutNeuron = ((imgDimension[0]-fiterDimension[0])/strideLength+1,
                       (imgDimension[1]-fiterDimension[1])/strideLength+1
                       )
    fLayerOutNeuron = 500
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

    matched,total,matchPercent = evaluator.accuracy(imageList, noBB, regionProposedArr, predScoreMatrixArr,imageObjBB,imageObjClass,noTopClass,noTopScoredRegionPerImg,overlapRatio)

    print "noTopScoredRegionPerImg : ",noTopScoredRegionPerImg
    print 'Matched : ',matched
    print 'Total : ',total
    print 'Accuracy : ',matchPercent,"%"

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

    

    
