import numpy as np
import time
import cv2
#import ImageFont, ImageDraw
import copy

vocClasses = ['person','bird', 'cat', 'cow', 'dog',
               'horse', 'sheep', 'aeroplane','bicycle', 'boat',
               'bus', 'car', 'motorbike', 'train', 'bottle',
               'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

cifarClasses = ['airplane','automobile','bird','cat','deer',
                'dog','frog','horse','ship','truck']

vocTrainImgPath = 'images'
vocTrainAnnPath = '../datasets/VOC2012trainval/VOC2012/Annotations'
vocTestImgPath = '../datasets/VOC2012test/VOC2012/JPEGImages'
vocTestAnnPath = '../datasets/VOC2012test/VOC2012/Annotations'



def nTopPredictions(predScoreMatrix,noTopClass):
    '''horizontal pruning'''
    
    cifarArray = np.asarray(cifarClasses)
    print "predScoreMatrix.shape:",predScoreMatrix.shape
    #raw_input("WAIT0:")
    nTopPred = np.argsort(predScoreMatrix)[:,::-1][:,:noTopClass]
    nTopScore = np.sort(predScoreMatrix)[:,::-1][:,:noTopClass]

    return nTopScore, nTopPred, cifarArray[nTopPred]


    
def getMatch(imageList, predBB, predOP,actualBB, actualOP):

    '''print "pred BB:",predBB         #list of lists of tuples
    print "pred OP:",predOP         #list of lists of int
    print "actualBB:",actualBB      #list of lists of tuples
    print "actualOP:",actualOP      #list of lists of int'''
    print "Lengths :",len(imageList),len(predBB),len(predOP),len(actualBB),len(actualOP)
    raw_input("getMatch():")
    if len(predOP)!=len(actualOP):
        print "ERROR NO OF IMAGES NOT MATCHED"
    noImage = len(actualOP)
    noActualClass = sum([len(i) for i in actualOP])
    print noActualClass
    totalMatch = 0
    for (act,pred) in zip(actualOP,predOP):
        eachImageMatch = 0
        for i in act:
            if i in pred:
                del pred[pred.index(i)]
                eachImageMatch += 1
        totalMatch += eachImageMatch

    return totalMatch,noActualClass,round((totalMatch*100/float(noActualClass)),2)


def overlap(aBB,pBB):

    areaActual = (aBB[2] - aBB[0])*(aBB[3] - aBB[1])
    areaPredicted =  (pBB[2] - pBB[0])*(pBB[3] - pBB[1])
    xintersect = max(0,(min(aBB[2],pBB[2]) - max(aBB[0],pBB[0])))
    yintersect = max(0,(min(aBB[3],pBB[3]) - max(aBB[1],pBB[1])))
    areaIntersect = xintersect * yintersect
    areaUnion = areaActual + areaPredicted - areaIntersect
    overlap = areaIntersect / float(areaUnion)

    #print aBB,pBB
    #print areaActual,xintersect,yintersect,areaPredicted,areaIntersect,areaUnion,overlap
    return overlap
    

def getMatch2(imageList, predBB, predOP,actualBB, actualOP, predConfiScore, overlapRatio):

    '''print "pred BB:",predBB         #list of lists of tuples
    print "pred OP:",predOP         #list of lists of int
    print "actualBB:",actualBB      #list of lists of tuples
    print "actualOP:",actualOP      #list of lists of int'''
    print "Lengths :",len(imageList),len(predBB),len(predOP),len(actualBB),len(actualOP), len(predConfiScore)
    #raw_input("getMatch():")
    if len(predOP)!=len(actualOP):
        print "ERROR NO OF IMAGES NOT MATCHED"
    noImage = len(actualOP)
    noActualClass = sum([len(i) for i in actualOP])
    #print noActualClass
    totalMatch = 0
    for (act,pred,predB,image,score) in zip(actualOP,predOP,predBB,imageList,predConfiScore):
        #print image
        '''print act,pred,actB,predB
        raw_input("loop():")'''
        img2 = cv2.imread(vocTrainImgPath+"/"+image)        
        eachImageMatch = 0
        #for (i,j) in zip(act,actB):
        for i in act:
            print "i:",i
            print "pred",pred
            #raw_input("loop2():")
            for k in xrange(len(pred)):
                p=predB[k]
                px1=int(float(p[0]))
                py1=int(float(p[1]))
                px2=int(float(p[2]))
                py2=int(float(p[3]))
                s=round(float(score[k][0]),2)
                if i == pred[k]:
                    #print "k:",k
                    #print pred[k],predB[k]
                    
                    #print s,s[0],float(s[0]),ss
                    #raw_input("JJJJ:")
                    ov = 1
                    #raw_input("if():") 
                    if ov > overlapRatio :
                        newov = round(ov,2)
                        del pred[k]     #req so that next pred class do not find the same region 
                        del predB[k]
                        del score[k]
                        eachImageMatch += 1
                        #draw actual BB
                        #cv2.rectangle(img2,(int(j[0]),int(j[1])),(int(j[2]),int(j[3])),(255,0,0),1)
                        #draw predicted BB

                        cv2.rectangle(img2,(px1,py1),(px2,py2),(0,255,0),1)
                        cv2.putText(img2,str(s),(px1+2,py1+10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1,False)
                        cv2.putText(img2,str(cifarClasses[i]),(px2-60,py2+15),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1,False)
                
                        break
        for k in xrange(len(pred)):
            p=predB[k]
            px1=int(float(p[0]))
            py1=int(float(p[1]))
            px2=int(float(p[2]))
            py2=int(float(p[3]))
            s=round(float(score[k][0]),2)
            cv2.rectangle(img2,(px1,py1),(px2,py2),(0,0,255),1)
            cv2.putText(img2,str(s),(px1+2,py1+10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1,False)
            cv2.putText(img2,str(cifarClasses[pred[k]]),(px2-60,py2+15),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1,False)
                        
        if eachImageMatch == 0:
            cv2.imwrite('NotIdentified/'+image,img2)
        else:    
            cv2.imwrite('FinalOP/'+image,img2)
        totalMatch += eachImageMatch

    return totalMatch,noActualClass,round((totalMatch*100/float(noActualClass)),2)


    
def accuracy(imageList, noBB, regionProposed, predScoreMatrix, imageObjBB,imageObjClass,noTopClass,noTopScoredRegionPerImg,overlapRatio):


    #gets n top class predicted horizontal pruning
    nTopScore, nTopPred, nTopPredClass = nTopPredictions(predScoreMatrix,noTopClass)
    #nTopPred not required

    print "regionProposed.shape:",regionProposed.shape
    print "nTopScore.shape:",nTopScore.shape
    print "nTopPredClass.shape:",nTopPredClass.shape

    region = [tuple(img) for img in regionProposed]

    #print regionProposed
    print "region length :",len(region)


    #raw_input("STOP0")
    
    result = np.hstack((nTopScore,nTopPredClass,region))
    #print result
    print result.shape
    #raw_input("STOP")

    startBB = 0
    endBB = 0

    #noBB : my predicted no of bounding boxes
    predOP = [] #total o/p class for all images list of list
    predBB = []
    predConfiScore=[]
    #for Each image find top n regions based on score and draw them 
    for i in xrange(len(noBB)):
        '''print "Image:",imageList[i]
        print noBB[i]'''
        startBB = endBB
        endBB = endBB+noBB[i]
        x = result[startBB:endBB,:]
        #print "x:",x
        #for each image sort the regions according to the first predicted score
        #z is 10 cols : top 3 score + top 3 prediction+ BB 4 axes
        z = x[np.argsort(x[:,0])[::-1]][:noTopScoredRegionPerImg]
        print "z:",z
        print "Z shape:",z.shape
        #raw_input("EWERE:")
        #predClass = z[:,3:4]
        #print 'predClass:',predClass
        predClass=[]
        predReg = []
        predScore = []
        for eachRegClass in z[:,3:4]:
            #print eachRegClass[0],str(eachRegClass[0]) 
            predClass.append(cifarClasses.index(eachRegClass[0]))
        for eachReg in z[:,6:10]:
            #print eachReg,tuple(eachReg)
            predReg.append(tuple(eachReg))
            #raw_input("SsasasTOP")
        for eachScore in z[:,0:1]:
            predScore.append(eachScore)
            
        predOP.append(predClass)
        predBB.append(predReg)
        predConfiScore.append(predScore)
        #print predOP
        #print predReg

        #raw_input("SsasasTOP")
        

        imgToBB = cv2.imread(vocTrainImgPath+"/"+imageList[i])
        for j in z:
            x1 = int(float(j[6]))
            y1 = int(float(j[7]))
            x2 = int(float(j[8]))
            y2 = int(float(j[9]))
            cv2.rectangle(imgToBB,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(imgToBB,str(j[3]),(x1,y2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,False)
        cv2.imwrite('BoundedRegion/'+imageList[i],imgToBB)

        #raw_input("STOsasP")

    print len(predOP)
    print len(imageObjClass)
    #raw_input("STOsasP")
    print predOP
    predOPOrig = copy.deepcopy(predOP)
    matched,total,matchPercent = getMatch2(imageList, predBB, predOP,imageObjBB, imageObjClass, predConfiScore,overlapRatio)    
        
    return matched,total,matchPercent,predOPOrig
