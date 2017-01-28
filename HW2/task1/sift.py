import sys
import cv2
import numpy as np
from random import choice
from scipy.special import *
from scipy import *
from scipy import linalg

def getDescriptorKP(img, file_name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray, None) 
    img = cv2.drawKeypoints(gray,kp)
    cv2.imwrite('keypoints_'+file_name,img)
    return kp, des
 

def findMatches(kp1, des1, kp2, des2):
    matches = []
    for i in range(len(kp1)):
        first = sys.maxint
        first_index = -1
        second = sys.maxint
        second_index = -2
        for j in range(len(kp2)):
            diff = des1[i]-des2[j]
	    dis = sum(diff**2)
	    
            if dis<first:
                second = first
                second_index = first_index
                first = dis
                first_index = j
            elif dis<second:
                second = dis
                second_index = j
        if first*1.0 / second*1.0 < 0.8:
            matches.append([kp1[i].pt, kp2[first_index].pt])
    print "The number of matches is",len(matches)
    return matches
    

def showMatches(img1, img2, matches, file_name):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    if rows1 < rows2:
        img1 = np.concatenate((img1, np.zeros((rows2 - rows1, img1.shape[1],3))), axis = 0)
    else:
        img2 = np.concatenate((img2, np.zeros((rows1 - rows2, img2.shape[1],3))), axis = 0)
    img3 = np.concatenate((img1, img2), axis = 1)
    for mat in matches:
        cv2.line(img3, (int(mat[0][0]),int(mat[0][1])), (int(mat[1][0])+cols1,int(mat[1][1])), (255, 0, 0), 1)
    cv2.imwrite(file_name, img3)


def affineMatches(matches, error, good_model_num, is_homograph):
    newMatches = []
    transformation = ransac(matches, error, good_model_num, is_homograph)
    for mat in matches:
        vector = list(mat[0])
        vector.append(1) 
        vector2 = np.dot(transformation,vector)
        if (abs(1.0 - vector2[0] / mat[1][0]) < 0.1) and (abs(1.0 - vector2[1] / mat[1][1]) < 0.1):
            newMatches.append(mat)
    print "The number of inlier matches is", len(newMatches)
    return newMatches, transformation


def Haffine_from_points(fp, tp):
    """ find H, affine transformation, such that 
        tp is affine transf of fp"""

    if fp.shape != tp.shape:
        raise RuntimeError, 'number of points do not match'

    #condition points
    #-from points-
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1]) 
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = dot(C1,fp)

    #-to points-
    m = mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = dot(C2,tp)

    #conditioned points have mean zero, so translation is zero
    A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = linalg.svd(A.T)

    #create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1) 
    H = vstack((tmp2,[0,0,1]))

    #decondition
    H = dot(linalg.inv(C2),dot(H,C1))

    return H / H[2][2]


def ransac(points_list, error, good_model_num, is_homograph):
    '''
        This function uses RANSAC algorithm to estimate the
        shift and rotation between the two given images
    '''
    
    model_error = 255
    model_H = None
    
    if is_homograph:
        error+=10
    for i in range(300):
        consensus_set = []
        points_list_temp = copy(points_list).tolist()
        # Randomly select 3 points
        for j in range(3):
            temp = choice(points_list_temp)
            consensus_set.append(temp)
            points_list_temp.remove(temp)
        
        # Calculate the homography matrix from the 3 points
        
        fp0 = []
        fp1 = []
        fp2 = []
        
        tp0 = []
        tp1 = []
        tp2 = []
        for line in consensus_set:
        
            fp0.append(line[0][0])
            fp1.append(line[0][1])
            fp2.append(1)
            
            tp0.append(line[1][0])
            tp1.append(line[1][1])
            tp2.append(1)
            
        fp = array([fp0, fp1, fp2])
        tp = array([tp0, tp1, tp2])
        
        H = []
        if (is_homograph):
            H = H_from_points(fp, tp)
        else:
            H = Haffine_from_points(fp, tp)
                            
        # Transform the second image
        # imtemp = transform_im(im2, [-xshift, -yshift], -theta)
        # Check if the other points fit this model

        for p in points_list_temp:
            x1, y1 = p[0]
            x2, y2 = p[1]

            A = array([x1, y1, 1]).reshape(3,1)
            B = array([x2, y2, 1]).reshape(3,1)
            
            out = B - dot(H, A)
            dist_err = hypot(out[0][0], out[1][0])
            if dist_err < error:
                consensus_set.append(p)            
            

        # Check how well is our speculated model
        if len(consensus_set) >= good_model_num:
            dists = []
            for p in consensus_set:
                x0, y0 = p[0]
                x1, y1 = p[1]
                
                A = array([x0, y0, 1]).reshape(3,1)
                B = array([x1, y1, 1]).reshape(3,1)
                
                out = B - dot(H, A)
                dist_err = hypot(out[0][0], out[1][0])
                dists.append(dist_err)
            if (max(dists) < error) and (max(dists) < model_error):
                model_error = max(dists)
                model_H = H
    print "Transformation matrix is : "
    print model_H
    return model_H                 

                 
def H_from_points(fp,tp):
    """ Find homography H, such that fp is mapped to tp
        using the linear DLT method. Points are conditioned
        automatically. """
    
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
        
    # condition points (important for numerical reasons)
    # --from points--
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1]) 
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = dot(C1,fp)
    
    # --to points--
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = dot(C2,tp)
    
    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = zeros((2*nbr_correspondences,9))
    for i in range(nbr_correspondences):        
        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]
    
    U,S,V = linalg.svd(A)
    H = V[8].reshape((3,3))    
    
    # decondition
    H = dot(linalg.inv(C2),dot(H,C1))
    
    # normalize and return
    return H / H[2,2]

 
def alignImages(img1, img2, transformation, file_name):
    img3 = np.zeros([img2.shape[0],img2.shape[1],3])
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            vector = [i,j,1]
            vector2 = np.dot(transformation, vector)
            if (vector2[0]<=img2.shape[0] and vector2[1]<=img2.shape[1] and vector2[0]>=0 and vector2[1]>=0):
                img3[vector2[0]][vector2[1]] = img1[i][j]
    cv2.imwrite(file_name+'_warped.jpg', img3)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            img3[i][j][0] = img2[i][j][0]
    cv2.imwrite(file_name+'_merged.jpg', img3) 


def measureAlignment(matches, transformation):
    distance = 0.0
    for mat in matches:
        vector = list(mat[0])
        vector.append(1) 
        vector2 = np.dot(transformation,vector)
        distance += abs(2.0 - vector2[0] / mat[1][0] - vector2[1] / mat[1][1])/2.0
    print "Average distance of alignments is:",distance/len(matches)
    return distance/len(matches)

 
def test(input1, input2, error, good_model_num):
    image1 = input1+'.png'
    image2 = input2+'.jpg'

    img1 = cv2.imread(image1)
    kp1, des1 = getDescriptorKP(img1, image1)
    
    img2 = cv2.imread(image2)
    kp2, des2 = getDescriptorKP(img2, image2)

    matches = findMatches(kp1, des1, kp2, des2)

    showMatches(img1, img2, matches, 'before_ransac_'+input1+'_'+image2)

    new_matches,transformation = affineMatches(matches, error, good_model_num, 0)

    showMatches(img1, img2, new_matches, 'after_ransac_'+input1+'_'+image2)

    alignImages(img1, img2, transformation, input1+'_'+input2)

    measureAlignment(new_matches, transformation)
    

    new_homo_matches, homo_transformation = affineMatches(matches, error, good_model_num, 1)
    
    showMatches(img1, img2, new_homo_matches, 'homo_after_ransac_'+input1+'_'+image2)
 
    alignImages(img1, img2, homo_transformation,'homo_'+input1+'_'+input2)

    measureAlignment(new_homo_matches, homo_transformation)

if __name__ == "__main__":
    #test('StopSign1', 'StopSign2', 20, 10)
    test('template','image3',200,10)
    #test('StopSign1', 'StopSign3', 40, 10)
    #test('StopSign1', 'StopSign4', 200, 15)

