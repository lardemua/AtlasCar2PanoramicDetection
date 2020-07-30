#!/usr/bin/env python2

from __future__ import print_function
import numpy as np
import argparse
import roslib
from panorama import Stitcher
#roslib.load_manifest('my_package')
import sys
import json
import os
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError




class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub_center = rospy.Subscriber("/center_camera/image_raw",Image,self.callback_center_image)
        self.image_sub_left = rospy.Subscriber("/left_camera/image_raw",Image,self.callback_left_image)
        self.image_sub_right = rospy.Subscriber("/right_camera/image_raw",Image,self.callback_right_image)

        self.image_sub_center_seg = rospy.Subscriber("/segnet/center_overlay",Image,self.callback_center_image_seg)
        self.image_sub_left_seg = rospy.Subscriber("/segnet/left_overlay",Image,self.callback_left_image_seg)
        self.image_sub_right_seg = rospy.Subscriber("/segnet/right_overlay",Image,self.callback_right_image_seg)
        
        self.left_image = None
        self.center_image = None
        self.right_image = None

        self.left_image_seg = None
        self.center_image_seg = None
        self.right_image_seg = None
    
    def callback_center_image(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.center_image = cv_image
        except CvBridgeError as e:
            print(e)
       
    
    def callback_left_image(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
            self.left_image = cv_image
        except CvBridgeError as e:
            print(e)
       
    
    def callback_right_image(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
            self.right_image = cv_image
        except CvBridgeError as e:
            print(e)

    def callback_center_image_seg(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.center_image_seg = cv_image
        except CvBridgeError as e:
            print(e)
       
    
    def callback_left_image_seg(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
            self.left_image_seg = cv_image
        except CvBridgeError as e:
            print(e)
       
    
    def callback_right_image_seg(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
            self.right_image_seg = cv_image
        except CvBridgeError as e:
            print(e)
       
    
    def clean_images(self):
        self.left_image = None
        self.center_image = None
        self.right_image = None

        self.left_image_seg = None
        self.center_image_seg = None
        self.right_image_seg = None



def pub_panorama(panorama):
    panorama_pub = rospy.Publisher("panoramic_image",Image,queue_size=1)
    bridge = CvBridge()
    try:
      panorama_pub.publish(bridge.cv2_to_imgmsg(panorama, "bgr8"))
    except CvBridgeError as e:
      print(e)

def pub_panorama_seg(panorama_seg):
    panorama_pub_seg = rospy.Publisher("panoramic_image_overlays",Image,queue_size=1)
    bridge = CvBridge()
    try:
      panorama_pub_seg.publish(bridge.cv2_to_imgmsg(panorama_seg, "bgr8"))
    except CvBridgeError as e:
      print(e)


        

def main(args):

    Pano = 1;
    rospy.init_node('panorama', anonymous=True)
    ic = image_converter()
    stitcher = Stitcher()
    M_left_center = None
    M_right_center = None


    while True:
        try:
            # if ic.left_image is not None and ic.center_image is not None and ic.right_image is not None and M_left_center is None and M_center_right is None:
            #     print("Cheguei aqui!")
            #     (M_left_center, M_center_right) = stitcher.transformationsCalculator([ic.left_image, ic.center_image, ic.right_image], ratio=0.8, reprojThresh=4.0)    and ic.left_image_seg is not None and ic.center_image_seg is not None and ic.right_image_seg is not None
            if ic.left_image is not None and ic.center_image is not None and ic.right_image is not None:
                
                if M_left_center is None and M_right_center is None:
                    (M_left_center, M_right_center) = stitcher.transformationsCalculator([ic.left_image, ic.center_image, ic.right_image], ratio=0.8, reprojThresh=4.0)


            
            if ic.left_image is not None and ic.center_image is not None and ic.right_image is not None:
                if Pano == 0:
                    
                    if M_left_center is not None and M_right_center is not None:
                      
                        result = stitcher.stitch([ic.left_image,ic.center_image, ic.right_image], M_left_center, M_right_center, ratio=0.8, reprojThresh=4.0)

                        if result is None:
                                print("There was an error in the stitching procedure")
                        else:
                            pub_panorama(result)
                            ic.clean_images()
                    else:
                        print("Nao foram calculadas as matrizes de transformacao")
            else:
                continue

            if ic.left_image_seg is not None and ic.center_image_seg is not None and ic.right_image_seg is not None:
                if Pano == 1:

                    if M_left_center is not None and M_right_center is not None:
                    
                        result = stitcher.stitch([ic.left_image_seg,ic.center_image_seg, ic.right_image_seg], M_left_center, M_right_center, ratio=0.8, reprojThresh=4.0)

                        if result is None:
                                print("There was an error in the stitching procedure")
                        else:
                            pub_panorama_seg(result)
                            ic.clean_images()
                    else:
                        print("Nao foram calculadas as matrizes de transformacao")

            else:
                continue
        except KeyboardInterrupt:
            print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)





















