import numpy as np
import cPickle
import time
from os import walk
import xml.etree.ElementTree as et

import A_A_selectiveSearchFilter as ssf



vocClasses = ['person','bird', 'cat', 'cow', 'dog',
               'horse', 'sheep', 'aeroplane','bicycle', 'boat',
               'bus', 'car', 'motorbike', 'train', 'bottle',
               'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

cifarClasses = ['airplane','automobile','bird','cat','deer',
                'dog','frog','horse','ship','truck']

path = 'images'
vocTrainAnnPath = '../datasets/VOC2012trainval/VOC2012/Annotations'
#vocTestImgPath = '../datasets/VOC2012test/VOC2012/JPEGImages'
#vocTestAnnPath = '../datasets/VOC2012test/VOC2012/Annotations'


'''voc_cifar_map = [('aeroplane','airplane'),
                 ('car','automobile'),
                 ('bird','bird'),
                 ('cat','cat'),
                 ('dog','dog'),
                 ('horse','horse'),
                 ('boat','ship'),
                 ('bus','truck')]'''

voc_cifar_map = {'aeroplane':'airplane',
                 'car':'automobile',
                 'bird':'bird',
                 'cat':'cat',
                 'dog':'dog',
                 'horse':'horse',
                 'boat':'ship',
                 'bus':'truck'
                }

startTest = 4900              #tochange
endTest= 5400
testDtFileName = "test0.5.pkl"    #tochange

    
def getAnnotations(path=vocTrainAnnPath):

    #--------- GET LIST OF ALL FILES IN THE PATH ----------
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break

    print "No of Images :",len(files)
    print files
    raw_input("WAIT")
    
    #--------- PARSE ANNOTATION XMLS ----------
    imageDetails = []
    for f in files:        
        tree = et.parse(path+'/'+f)
        root = tree.getroot()
        noObject = 0
        listObj = []
        d={}    #for each file
        for child in root:
            #print child.tag,child.attrib,child.text
            if child.tag == "filename":
                d[child.tag]=child.text
                #d[child.tag]=child.text
                #print child.text
            if child.tag == "size":
                d[child.tag]={child[0].tag:int(child[0].text),
                              child[1].tag:int(child[1].text),
                              child[2].tag:int(child[2].text)}
            if child.tag == "object":
                x={} #for each object
                noObject+=1
                x["Objname"]="Object"+str(noObject)
                
                for subchild in child:
                    if subchild.tag == "bndbox":
                        x[subchild.tag]={subchild[0].tag:float(subchild[0].text),
                                         subchild[1].tag:float(subchild[1].text),
                                         subchild[2].tag:float(subchild[2].text),
                                         subchild[3].tag:float(subchild[3].text)}
                    
                    if subchild.tag == "name":
                        #x[subchild.tag] = vocClasses.index(subchild.text)
                        x[subchild.tag] = subchild.text #change for cifar-voc
                    else:
                        continue
                listObj.append(x)

            else:
                continue
        d['object'] = listObj
        d['noObjects'] = noObject 

        imageDetails.append(d)
        #print imageDetails[len(imageDetails)-1]
        #raw_input("WAIT:")
        
    #print len(imageDetails)
    print imageDetails[1]
    #print voc_cifar_map

    # ------------ GET IMAGE NAME AND OBJECT CLASSES -------------


    imageList = []
    imageObjBB = []
    imageObjno = []
    imageObjClass = []
    for image in imageDetails[startTest:endTest]:
        imageList.append(image['filename'])
        imageObjno.append(image['noObjects'])
        BB=[]
        oCls = []
        for eachObject in image['object']:
            #print eachObject,"\n"
            if eachObject['name'] in voc_cifar_map.keys() :
                BB.append((eachObject['bndbox']['xmin'],eachObject['bndbox']['ymin'],eachObject['bndbox']['xmax'],eachObject['bndbox']['ymax']))
                cifarCls = voc_cifar_map[eachObject['name']]
                #print eachObject['name']
                #print cifarCls
                oCls.append(cifarClasses.index(cifarCls))
            #raw_input("WAITTT:")
        imageObjBB.append(BB)
        imageObjClass.append(oCls)
        '''print "\n",imageList
        print imageObjBB
        print imageObjno
        print imageObjClass'''
        #raw_input("WAITTT2:")
        
    

    return imageList,imageObjBB,imageObjno,imageObjClass

if __name__ == "__main__":


    eachBBsize = (32,32)
    imageList = ['multi1.jpg','multi2.jpg','multi3.jpg','multi4.jpg','multi5.jpg',
             'multi6.jpg','multi7.jpg','multi8.jpg','multi9.jpg','multi10.jpg']
    noImage = len(imageList)

    '''st = time.time()
    imageList,imageObjBB,imageObjno,imageObjClass = getAnnotations(vocTrainAnnPath)
    print "Time taken to Parse annotations :",round(time.time()-st)
    '''
    imageObjBB=[]
    imageObjno =[]
    imageObjClass = [[1,2,8],[1,2,8],[4,8,9],[2,6,7],[1,3,9],[1,8,9],[0,1,9],[0,1,9],[4,6,7],[0,2,2]]
    st = time.time()
    regionProposed,noBB,regionArray = ssf.getRegionProposals(path, imageList, eachBBsize)
    print "Time taken to propose regions :",round(time.time()-st)
    
    print len(regionProposed),len(noBB),len(regionArray)
    raw_input("WAIT:")

    #------------ LOADS PREPROCESSING PARAMS (ZCA PARAMS) -------------
    f = open("../zcaParams.pkl","rb")
    mean, zcawhite = cPickle.load(f)
    f.close()
    
    #print mean.shape                   #(3072,)
    #print zcawhite.shape               #(3072,3072)

    regionPixels = np.vstack(regionArray)   #list of np array to np array
    print regionPixels.shape
    print regionPixels

    #------------ TEST DATA NORMALIZED, WHITENNED, ROTATED ----------------
    regionPixels/=255.
    #print regionPixels
    regionPixels -= mean
    #print regionPixels
    regionPixels = np.dot(regionPixels,zcawhite.T)  
    print regionPixels.shape

    noRegionsTotal = regionPixels.shape[0]
    regionPixels=np.reshape(regionPixels,(noRegionsTotal,eachBBsize[0]*eachBBsize[1],3)) #(72,1024,3)
    regionPixels = np.transpose(regionPixels,(0,2,1))   #(72,3,1024)
    print regionPixels.shape
    testData = [x for x in regionPixels]
    print "--------------------------------------------"
    print len(imageObjClass)
    print len(testData)

    raw_input("TTTTTTT:")
    #------------ TEST DATA EXPORTED -----------------
    st = time.time()
    f = open(testDtFileName,"wb")
    cPickle.dump([imageList, noBB, regionProposed, testData, imageObjClass, imageObjBB, imageObjno],f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print "Time taken to dump :",round(time.time()-st)
    





    
