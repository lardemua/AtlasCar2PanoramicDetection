# ROS-panorama-package
This package is a set of ROS nodes that allows the creation of a panorama image based on 3 cameras and is a alternative for  [this package](https://github.com/rubendfcosta/AtlasCar2PanoramicDetection/tree/master/Panoramic-Image-Python).

### Prerequisites

What things that are needed:

* [ROS-melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) - The ROS version used.

Hardware:
* [3 USB Cameras](https://www.logitech.com/pt-br/product/hd-webcam-c270) - Log


### Usage Instructions

After all the prerequisites are installed, you have to connect the 3 USB cameras to your computer and edit the **camera_panoramic.launch** file and set yours devices ID's. 

To launch the entire system run:
```
roslaunch panorama_imageCPP camera_panoramic.launch
```



