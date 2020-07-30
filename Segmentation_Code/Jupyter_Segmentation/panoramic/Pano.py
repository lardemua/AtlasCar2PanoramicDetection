class Stitcher:
    def __init__(self):
        self.cachedHlc = None
        self.cachedHrc = None
        
    def stitch(self, images, Masks, ratio=0.8, reprojThresh=4.0):
        (image_left, image_center, image_right) = images
        (Maskleft_image, Maskcenter_image, Maskright_image) = Masks
        
        
        ##############EXECUTED ONLY ONE TIME##############
        if self.cachedHlc is None or self.cachedHrc is None:

            (kpsLeft, featuresLeft) = self.detectAndDescribe(image_left)
            (kpsCenter, featuresCenter) = self.detectAndDescribe(image_center)
            (kpsRight, featuresRight) = self.detectAndDescribe(image_right)


            M_left_center = self.matchKeypoints(kpsLeft, kpsCenter,featuresLeft, featuresCenter, ratio, reprojThresh)
            M_right_center = self.matchKeypoints(kpsRight, kpsCenter,featuresRight, featuresCenter, ratio, reprojThresh)
            
            
            #if M_left_center is None or M_right_center is None:
			#	return None
            self.cachedHlc = M_left_center[1]
            self.cachedHrc = M_right_center[1]
        
        ##################################################

        import kornia 
        
        result_width = 3200
        T = np.array([[1.0, 0.0, (result_width/2)-(images[0].shape[1]/2)],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

        ## warp the original image by the found transform
        #data_warp: torch.tensor = kornia.warp_perspective(data.float(), M, dsize=(h, w))


        transformations = [self.cachedHlc, np.identity(3, dtype=np.float32), self.cachedHrc]
        result = np.zeros((Masks[0].shape[0],result_width,3)).astype(np.float32)
        weights = np.zeros_like(result)
        
        for i in range(len(Masks)):
            warp = kornia.warp_perspective(Masks[i], 
                                       np.dot(T,transformations[i]), 
                                       (result_width, images[i].shape[0])).astype(np.float32)
            weight = kornia.warp_perspective(np.ones_like(Masks[i]), 
                                       np.dot(T,transformations[i]), 
                                       (result_width, images[i].shape[0])).astype(np.float32)
            result =  cv2.addWeighted(result,1.0,warp,1.0,0.0)
            weights = cv2.addWeighted(weights,1.0,weight,1.0,0.0)

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
        if len(matches) > 25:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
            return (matches, H, status)
        return None
