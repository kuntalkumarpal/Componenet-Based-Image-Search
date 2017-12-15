import numpy as np
import cv2

#  Malisiewicz et al.
def nonMaximumSuppression(boundingBoxes, overlapThresh):
    
    if boundingBoxes.shape[0] == 0 :
        print "NO BOUNDING BOX"
        return []

    boundingBoxes = boundingBoxes.astype("float32")

    selectedBB = []

    x1 = boundingBoxes[:,0]
    y1 = boundingBoxes[:,1]
    x2 = boundingBoxes[:,2]
    y2 = boundingBoxes[:,3]

    '''print x1
    print y1
    print x2
    print y2'''

    area = (x2-x1+1)*(y2-y1+1)
    #print area

    idxs = np.argsort(y2)
    #print idxs
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        selectedBB.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w=np.maximum(0, xx2-xx1+1)
        h=np.maximum(0, yy2-yy1+1)

        overlap = (w*h) / area [idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap>overlapThresh)[0])))

    return boundingBoxes[selectedBB]


if __name__ == '__main__' :
    
    listOfBB = np.array([(316, 305, 429, 391),
(332, 186, 418, 276),
(235, 83, 382, 213),
(280, 105, 563, 396),
(369, 123, 507, 208),
(288, 292, 429, 391),
(280, 105, 559, 396),
#(0, 0, 599, 399),
(280, 105, 548, 395),
(316, 305, 423, 385),
(235, 83, 507, 282),
(235, 83, 507, 276)])
    print listOfBB
    thresh = 0.5
    finalBB = nonMaximumSuppression(listOfBB,thresh)
    print finalBB

    image = cv2.imread('image2.jpg')
    img = image.copy()
    for (sx,sy,ex,ey) in finalBB:
        print  (sx,sy,ex,ey) 
        cv2.rectangle(img,(sx,sy),(ex,ey),(0,255,0),2)

    #cv2.imshow("After NMS", img)

    cv2.imwrite('BBimage2.jpg',img)
