import torch
import kornia
import cv2
import numpy as np
import imutils
from glob import glob
import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from time import time

def create_cityscapes_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[:19, :] = np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])
    return ListedColormap(colormap / 255., N=256)


class Stitcher:
    def __init__(self):
        self.cachedHlc = None
        self.cachedHrc = None
        
    def stitch(self,Masks,result_width,M_left_center,M_center_right):
        (Maskleft_image, Maskcenter_image, Maskright_image) = Masks
        
        self.cachedHlc = M_left_center
        self.cachedHrc = M_center_right
        
        #result_width = 3200
        T = np.array([[1.0, 0.0, (result_width/2)-(Masks[0].shape[3]/2)],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]).astype(dtype=np.float32)
        

        transformations = [self.cachedHlc, np.identity(3, dtype=np.float32), self.cachedHrc]

        result = 0
        weights = 1e-6
        
        for i in range(len(Masks)):
            warp = cv2.warpPerspective(images[i], 
                                       np.dot(T,transformations[i]), 
                                       (result_width, images[i].shape[0])).astype(np.float32)
            weight = cv2.warpPerspective(np.ones_like(images[i]), 
                                       np.dot(T,transformations[i]), 
                                       (result_width, images[i].shape[0])).astype(np.float32)
            result =  cv2.addWeighted(result,1.0,warp,1.0,0.0)
            weights = cv2.addWeighted(weights,1.0,weight,1.0,0.0)

        return result / weights
        
    def transformationsCalculator(self,images,ratio=0.8, reprojThresh=4.0):
        (image_left, image_center, image_right) = images
        
        (kpsLeft, featuresLeft) = self.detectAndDescribe(image_left)
        (kpsCenter, featuresCenter) = self.detectAndDescribe(image_center)
        (kpsRight, featuresRight) = self.detectAndDescribe(image_right)

        if kpsLeft is None or kpsCenter is None or kpsRight is None:
            print("It was not possible to extract the keypoints")
            return None

        M_left_center = self.matchKeypoints(kpsLeft, kpsCenter,featuresLeft, featuresCenter, ratio, reprojThresh)
        M_right_center = self.matchKeypoints(kpsRight, kpsCenter,featuresRight, featuresCenter, ratio, reprojThresh)

        if M_left_center is None or M_right_center is None:
            print("Uma das matrizes nao foi calculada")
        return (M_left_center[1],M_right_center[1])
    
    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.ORB_create(nfeatures=1000)
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


def polygonGenerator(image,hullMode):
    import scipy
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    import numpy as np
    import matplotlib.pyplot as plt
    from shapely import geometry
    from descartes import PolygonPatch
    
    
    #select the coordinates where the edges where found and define [0,0] as point to close the loop
    if hullMode ==False:
        #Get coordinates where road was detected
        points = np.column_stack(np.where((image)==1))
    else:
        #Convert the image to grayscale & find edges
        img = np.uint8(np.float32(image))
        edges = cv2.Canny(img,3,3)
        coords = np.column_stack(np.where(edges==255))
        points = np.zeros((len(coords)+1, 2))
        points[:len(coords),:]= coords
        points[len(coords),0]= 736

    return points

def polygonStitcher(Points,img_height,img_width,M_left_center,M_center_right,result_width,hullMode,Lines=False):
    import cv2
    import numpy as np
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    from shapely import geometry
    from descartes import PolygonPatch

    (PointsE, PointsM, PointsD) = Points
    
    pointsE[:,0]= img_height-pointsE[:,0]
    pointsM[:,0]= img_height-pointsM[:,0]
    pointsD[:,0]= img_height-pointsD[:,0]
    
    
    pointsD[:,[0, 1]] = pointsD[:,[1, 0]]
    pointsM[:,[0, 1]] = pointsM[:,[1, 0]]
    pointsE[:,[0, 1]] = pointsE[:,[1, 0]]
    d = np.array([pointsD.astype('float32')])
    m = np.array([pointsM.astype('float32')])
    e = np.array([pointsE.astype('float32')])

    T = np.array([[1.0, 0.0, (result_width/2)-(img_width/2)],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]]).astype(dtype=np.float32)


    pointsOutD = cv2.perspectiveTransform(d, np.dot(T,M_center_right))
    pointsOutM = cv2.perspectiveTransform(m, T)
    pointsOutE = cv2.perspectiveTransform(e, np.dot(T,M_left_center))
    
    pointsOutE = pointsOutE[-1,:,:]
    pointsOutM = pointsOutM[-1,:,:]
    pointsOutD = pointsOutD[-1,:,:]
    
    imagemPoints = np.zeros([img_height,result_width])
    
    pointList=np.concatenate((pointsOutE,pointsOutM,pointsOutD))
    
    if hullMode ==True:
        pointListPoly= []
        hull= ConvexHull(pointList)
        

        if Lines == True:
            pointsTeste = np.zeros((len(pointList[hull.vertices[:],0])+1, 2))
            pointsTeste[:len(pointList[hull.vertices[:],0]),:]= pointList[hull.vertices[:],:]
            pointsTeste[len(pointList[hull.vertices[:],0]),:]=  pointList[hull.vertices[0],:]
            fig, ay = plt.subplots(figsize=(15, 15))
            #ay.set_title('Road Limits')
            ay.imshow(imagemPoints,cmap=cmap)
            ay.set_axis_off()  
            ay.plot(pointsTeste[:,0], pointsTeste[:,1], linestyle='solid', color='gainsboro', lw=1)
            xrange = [0, result_width]
            yrange = [0, img_height]
            ay.set_xlim(*xrange)
            ay.set_ylim(*yrange)
            ay.set_aspect(1)
            plt.savefig('Road-Limits.jpg')
        else:
            for i in range(len(hull.vertices[:])):
                p = geometry.Point(pointList[hull.vertices[i],1],pointList[hull.vertices[i],0])
                pointListPoly.append(p)

            poly = geometry.Polygon([[p.y, p.x] for p in pointListPoly])
            x,y = poly.exterior.xy
            ring_patch = PolygonPatch(poly)
            ring_patch.set_color([0.92,0.92,0.92])
            fig, az = plt.subplots(figsize=(15, 15))
            az.imshow(imagemPoints,cmap=cmap)
            az.add_patch(ring_patch)
            #az.set_title('Road Polygon')
            xrange = [0, result_width]
            yrange = [0, img_height]
            az.set_xlim(*xrange)
            az.set_ylim(*yrange)
            az.set_aspect(1)
            az.set_axis_off()
            plt.savefig('Road-Polygon.jpg')
        
    else:
        
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(imagemPoints,cmap='gray')
        ax.scatter(pointList[:,0], pointList[:,1], marker='.', color=[0.92,0.92,0.92],lw=1)
        #ax.set_title('Road Mask')
        xrange = [0, result_width]
        yrange = [0, img_height]
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        ax.set_aspect(1)
        ax.set_axis_off()
        plt.savefig('RoadMask.jpg')
     
    return pointList

cmap = LinearSegmentedColormap.from_list('mycmap', [(0 / 1, 'black'),
                                                    (1 / 1, [0.86,0.86,0.86])])
colormap = create_cityscapes_colormap()
colormap_gray = create_cityscapes_colormap()
torch.set_grad_enabled(False)

device = torch.device('cuda:2')
model = torch.jit.load('contextnet14_bdd100k_miou0.503.pth')
model = model.cpu()
#model = model.to(device).eval()

ID = 71
CAMADA = 19

imagesDireita = glob(f'imagens_novas/D-_{ID:04d}_Camada {CAMADA}.jpg')
imagesEsquerda = glob(f'imagens_novas/E-_{ID:04d}_Camada {CAMADA}.jpg')
imagesMeio = glob(f'imagens_novas/M-_{ID:04d}_Camada {CAMADA}.jpg')

imgsD = np.stack([ cv2.imread(f) for f in imagesDireita ])
imgsE = np.stack([ cv2.imread(f) for f in imagesEsquerda ])
imgsM = np.stack([ cv2.imread(f) for f in imagesMeio ])

imgsD = np.stack([ cv2.resize(img, (1280,736 )) for img in imgsD ])
imgsE= np.stack([ cv2.resize(img, (1280,736 )) for img in imgsE ])
imgsM = np.stack([ cv2.resize(img, (1280,736 )) for img in imgsM ])

imgsD = imgsD[..., ::-1]
imgsE = imgsE[..., ::-1]
imgsM = imgsM[..., ::-1]

imgsD = imgsD[:, :, ...]
imgsE = imgsE[:, :, ...]
imgsM = imgsM[:, :, ...]
    
left_image = imgsE[0]
center_image = imgsM[0]
right_image = imgsD[0]
        
result_width=2300

stitcher = Stitcher()

(M_left_center, M_center_right) =stitcher.transformationsCalculator([left_image,center_image, right_image],ratio=0.8, reprojThresh=4.0)

LI=torch.from_numpy(left_image.copy())
MI=torch.from_numpy(center_image.copy())
RI=torch.from_numpy(right_image.copy())
LI=LI[None, :, :]
LI=LI.permute(0,3, 1, 2)
MI=MI[None, :, :]
MI=MI.permute(0,3, 1, 2)
RI=RI[None, :, :]
RI=RI.permute(0,3, 1, 2)


start = time()
for i in range(1):
    result_width=2300
    if left_image is not None and center_image is not None and right_image is not None:
        start = time()
        resultMaskFinal = stitcher.stitch([LI.float(),MI.float(),RI.float()],result_width,M_left_center,M_center_right)
        end = time()
        record= end - start
        print(record)
        PanoramicImage=resultMaskFinal[-1,:,:,:].permute(1, 2,0)


        if resultMaskFinal is None:
            print("There was an error in the stitching procedure")
        else:
            #end = time()
            #record= end - start
            print(record)
            #fig, ax = plt.subplots(figsize=(15, 15))
            #ax.imshow(PanoramicImage/ 255.)
            #ax.set_axis_off()
            #plt.savefig('Pano-_0039_Camada 51.jpg', dpi=500)
    else:
        print("Falta as imagens!")
