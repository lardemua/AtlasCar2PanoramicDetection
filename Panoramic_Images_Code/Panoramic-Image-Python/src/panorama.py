#!/usr/bin/env python2

import cv2
import numpy as np
import imutils
import time

# input:images


class Stitcher:
    def __init__(self):
        self.cachedHlc = None
        self.cachedHrc = None
        
    def stitch(self, images, ratio=0.7, reprojThresh=4.0,showMatches=True):
        
        (image_left, image_center, image_right) = images
        
        
        
        if self.cachedHlc is None or self.cachedHrc is None:

            (kpsLeft, featuresLeft) = self.detectAndDescribe(image_left)
            (kpsCenter, featuresCenter) = self.detectAndDescribe(image_center)
            (kpsRight, featuresRight) = self.detectAndDescribe(image_right)


            M_left_center = self.matchKeypoints(kpsLeft, kpsCenter,featuresLeft, featuresCenter, ratio, reprojThresh)
            M_right_center = self.matchKeypoints(kpsRight, kpsCenter,featuresRight, featuresCenter, ratio, reprojThresh)
            
            if M_left_center is None or M_right_center is None:
			    return None
            self.cachedHlc = M_left_center[1]
            self.cachedHrc = M_right_center[1]

        result_width = 1920 
        T = np.array([[1.0, 0.0, (result_width/2)-(images[0].shape[1]/2)],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

        #so para ver tempo total
        start_time = time.time()

        transformations = [self.cachedHlc, np.identity(3, dtype=np.float32), self.cachedHrc]
        result = np.zeros((images[0].shape[0],result_width,3)).astype(np.float32)
        weights = np.zeros_like(result)

        for i in range(0,len(images)):
            warp = cv2.warpPerspective(images[i], 
                                       np.dot(T,transformations[i]), 
                                       (result_width, images[i].shape[0])).astype(np.float32)
            weight = cv2.warpPerspective(np.ones_like(images[i]), 
                                       np.dot(T,transformations[i]), 
                                       (result_width, images[i].shape[0])).astype(np.float32)

            result =  cv2.addWeighted(result,1.0,warp,1.0,0.0)
            weights = cv2.addWeighted(weights,1.0,weight,1.0,0.0)  
        print("\n\n\n--- %s seconds ---\n\n\n", (time.time() - start_time))
        return np.uint8(result / weights)
    







    















    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.ORB_create()
        kps = detector.detect(gray, None)
        (kps,features) = detector.compute(gray, kps)
        kps = np.float32([kp.pt for kp in kps])
        return (kps,features) 


    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(matches) > 15:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
            return (matches, H, status)
        return None




