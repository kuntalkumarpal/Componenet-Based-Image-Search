import numpy as np
import cv2
import selectivesearch as ss


def getRegionProposals(path, imageList, size):

    noBB = []               #list of no of BB of all images
    regionProposed = []     #list of all BB proposed for all images
    regionArray = []        #list of ndarray of BB proposed for all images
    
    noImage = len(imageList)
    ii = 1
    for image in imageList:
        print ii,":",image
        ii+=1
        img = cv2.imread(path+"/"+image)
        #print img.shape

        #  ---------------- PERFORM SELECTIVE SEARCH (EACH)---------------
        img_lbl, regions = ss.selective_search(img, scale=500, sigma=0.9, min_size=10)


        #  ---------------- FILTER REGIONS PROPOSED (EACH)----------------
        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 500 pixels
            if r['size'] < 500:
                continue            
            # distorted rects
            x, y, w, h = r['rect']
            #if w / h > 1.2 or h / w > 1.2:
            if ( w == 0 ) or (h == 0):
                continue
            if w / h > 2 or h / w > 2:
                continue
            if (img.shape[0]-w < 20) and (img.shape[0]-h < 20):
                continue
            candidates.add(r['rect'])


        #  ---------------- REGIONS PROPOSED (EACH)----------------
        regionProposedeach =[]
        img2 = img.copy()
        for (sx,sy,w,h) in candidates:
            regionProposedeach.append((sx,sy,sx+w,sy+h)) 
            cv2.rectangle(img2,(sx,sy),(sx+w,sy+h),(0,255,0),2)
            
        #draw rectangles on original image
        cv2.imwrite('Region/'+image,img2)

        noBBeach = len(regionProposedeach)
        print noBBeach



        ### ----------- EACH IMAGE NMS -------------- ###
        
        
        #  ---------------- GET REGION PIXELS (EACH)----------------
        cnt=1
        regionArrayeach = np.zeros((noBBeach, size[0]*size[1]*3))
        for region in regionProposedeach:
            imgBB = img[region[1]:region[3],region[0]:region[2]]
            modBB = cv2.resize(imgBB,size)                          #(32, 32, 3)
            #cv2.imwrite("BB/"+str(cnt)+image,modBB)
            modBB = np.reshape(modBB,(size[0]*size[1]*3))           #(1024, 3)
            regionArrayeach[cnt-1] = modBB
            cnt+=1


        #  ---------------- CREATING RETURNABLES ----------------        
        regionProposed.append(regionProposedeach)
        noBB.append(noBBeach)
        regionArray.append(regionArrayeach)

    '''print noBB
    print regionProposed
    print regionArray'''

    '''for i in regionArray:
        print i.shape'''

    
    return regionProposed,noBB,regionArray
