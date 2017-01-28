import sys
import cv2
import numpy as np
from random import choice
from scipy.special import *
from scipy import *
from scipy import linalg
from matplotlib import pyplot as plt
from PIL import Image
import os

def showMatches(img1, img2, src_pts, dst_pts, matchesMask, file_name):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
   
    if rows1 < rows2:
        img1 = np.concatenate((img1, np.zeros((rows2 - rows1, img1.shape[1],3))), axis = 0)
    else:
        img2 = np.concatenate((img2, np.zeros((rows1 - rows2, img2.shape[1],3))), axis = 0)
    img3 = np.concatenate((img1, img2), axis = 1)
    for i in range(len(matchesMask)):
        if (matchesMask[i]):
            cv2.line(img3, (int(src_pts[i][0][0]),int(src_pts[i][0][1])), (int(dst_pts[i][0][0])+cols1,int(dst_pts[i][0][1])), (255, 0, 0), 1)
    cv2.imwrite(file_name+'.jpg', img3)

def analyseImage(path):
    '''
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode 
    before processing all frames.
    '''
    im = Image.open(path)
    results = {
        'size': im.size,
        'mode': 'full',
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results
 
 
def processImage(path):
    '''
    Iterate the GIF, extracting each frame.
    '''
    mode = analyseImage(path)['mode']
    
    im = Image.open(path)
 
    i = 0
    p = im.getpalette()
    last_frame = im.convert('RGBA')
    
    try:
        while True:
            print "saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile)
            
            '''
            If the GIF uses local colour tables, each frame will have its own palette.
            If not, we need to apply the global palette to the new frame.
            '''
            if not im.getpalette():
                im.putpalette(p)
            
            new_frame = Image.new('RGBA', im.size)
            
            '''
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
            If so, we need to construct the new frame by pasting it on top of the preceding frames.
            '''
            if mode == 'partial':
                new_frame.paste(last_frame)
            
            new_frame.paste(im, (0,0), im.convert('RGBA'))
            new_frame.save('%s-%d.png' % (''.join(os.path.basename(path).split('.')[:-1]), i), 'PNG')
 
            i += 1
            last_frame = new_frame
            im.seek(im.tell() + 1)
    except EOFError:
        pass

def alignImages(img1, img2, transformation, file_name):

    img3 = img2
    point1 = np.dot(transformation, [0,0,1])
    point4 = np.dot(transformation, [img1.shape[0],img1.shape[1],1])

    point1[0] = max(point1[0],0)
    point1[0] = min(point1[0],img2.shape[0])
    point1[1] = max(point1[1],0)
    point1[1] = min(point1[1],img2.shape[1])

    point4[0] = max(point4[0],0)
    point4[0] = min(point4[0],img2.shape[0])
    point4[1] = max(point4[1],0)
    point4[1] = min(point4[1],img2.shape[1])
    cv2.rectangle(img3,(int(point1[0]),int(point1[1])),(int(point4[0]),int(point4[1])),(0, 255, 0))

    point1 = np.dot(transformation, [0,img1.shape[1],1])
    point4 = np.dot(transformation, [img1.shape[0],0,1])

    point1[0] = max(point1[0],0)
    point1[0] = min(point1[0],img2.shape[1])
    point1[1] = max(point1[1],0)
    point1[1] = min(point1[1],img2.shape[0])

    point4[0] = max(point4[0],0)
    point4[0] = min(point4[0],img2.shape[1])
    point4[1] = max(point4[1],0)
    point4[1] = min(point4[1],img2.shape[0])
    cv2.rectangle(img3,(int(point1[0]),int(point1[1])),(int(point4[0]),int(point4[1])),(0, 0, 255))
    cv2.imwrite(file_name+'rectangle.jpg', img3) 

def drawRetangle(img1, img2, src_pts, dst_pts, matchesMask,file_name):
    img3 = img2
    src_disx = 0
    src_disy = 0
    dst_disx = 0
    dst_disy = 0
    lastx = 0
    lastdx = 0
    lastdy = 0
    lasty = 0
    start = 0
    mindis = img2.shape[1]*img2.shape[0]
    minindex = start
    for i in range(len(src_pts)):
        if matchesMask[i]:
            lastx = src_pts[i][0][0]
            lasty = src_pts[i][0][1]
            lastdx = dst_pts[i][0][0]
            lastdy = dst_pts[i][0][1]
            start = i
            break
    for i in range(start+1, len(src_pts)):
        if matchesMask[i]:
            dis = (dst_pts[i][0][0]-lastdx)**2 + (dst_pts[i][0][1]-lastdy)**2
            if (dis<mindis):
                mindis = dis
                minindex = i
            src_disx = src_disx + abs(src_pts[i][0][0]-lastx)
            src_disy = src_disy + abs(src_pts[i][0][1]-lasty)
            dst_disx = dst_disx + abs(dst_pts[i][0][0]-lastdx)
            dst_disy = dst_disy + abs(dst_pts[i][0][1]-lastdy)
            lastx = src_pts[i][0][0]
            lasty = src_pts[i][0][1]
            lastdx = dst_pts[i][0][0]
            lastdy = dst_pts[i][0][1]


    ratiox = dst_disx / src_disx
    ratioy = dst_disy / src_disy

    # src_pts[minindex][0][0]+=sx1
    # src_pts[minindex][0][1]+=sy1
    # dst_pts[minindex][0][0]+=sx2
    # dst_pts[minindex][0][1]+=sy2
    point1 = [dst_pts[minindex][0][0] - src_pts[minindex][0][0] * ratiox, dst_pts[minindex][0][1] - src_pts[minindex][0][1] * ratioy]
    point4 = [dst_pts[minindex][0][0] + (img1.shape[0]-src_pts[minindex][0][0]) * ratiox, dst_pts[minindex][0][1] + (img1.shape[1]-src_pts[minindex][0][1]) * ratioy]
    '''print file_name
    print dst_pts[minindex][0]
    print src_pts[minindex][0]
    print ratiox
    print ratioy
    print point1
    print point4
    '''
    point1[0] = max(point1[0],0)
    point1[0] = min(point1[0],img2.shape[1])
    point1[1] = max(point1[1],0)
    point1[1] = min(point1[1],img2.shape[0])

    point4[0] = max(point4[0],0)
    point4[0] = min(point4[0],img2.shape[1])
    point4[1] = max(point4[1],0)
    point4[1] = min(point4[1],img2.shape[0])

    cv2.circle(img3, (int(dst_pts[minindex][0][0]),int(dst_pts[minindex][0][1])), 5, (0, 255, 0))
    cv2.circle(img3, (int(point1[0]),int(point1[1])), 5, (255, 0, 0))
    cv2.circle(img3, (int(point4[0]),int(point4[1])), 5, (255, 0, 0))
    cv2.rectangle(img3,(int(point1[0]),int(point1[1])),(int(point4[0]),int(point4[1])),(0, 0, 255),5)
    cv2.imwrite('rectangle_'+file_name+'.jpg', img3) 


def test(input1, input2):

    image1 = input1+'.jpg'
    image2 = input2+'.jpg'
    MIN_MATCH_COUNT = 0

    img1 = cv2.imread(image1)          # queryImage
    img2 = cv2.imread(image2) # trainImage

    sift = cv2.SIFT()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)     
    kp2, des2 = sift.detectAndCompute(img2,None)

    print input2    

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []

    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print len(good)
    

    if len(good)>MIN_MATCH_COUNT:
        
        
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
            matchesMask = mask.ravel().tolist()
            #xshowMatches(img1, img2, src_pts, dst_pts, matchesMask, 'ratiotest_'+input1+'_'+input2+'_'+str(num))
            
            #alignImages(img1, img2, M, input1+'_'+input2)
            drawRetangle(img1, img2, src_pts, dst_pts, matchesMask,input2)
            
        except Exception, e:
            print e
            pass
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    
if __name__ == "__main__":
    #test('StopSign1', 'StopSign2', 20, 10)
    
    test('template','image1')
    test('template','image2')
    test('template','image3')
    test('template','image4')
    test('template','image5')
    test('template','image6')
    test('template','image7')
    test('template','image8')
    test('template','image9')
    test('template','image10')
    test('template','image11')
    test('template','image12')
    test('template','lollipop-man')
    
