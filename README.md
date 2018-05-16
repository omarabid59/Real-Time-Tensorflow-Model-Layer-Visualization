# DIY Real Time Tensorflow Model Layer Visualization
This presents sample code to visualize some of the layers of a model loaded into tensorflow. We describe the process of choosing these layers and displaying them to screen.

## Overview
The image shows the real time visualization of the layers as the system is processing input from a webcam. We use the ssd_inception_v2_coco network from [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). The layers are hand picked and a random subset are plotted together.

![Sample Image Output during visualization of the layers](https://github.com/omarabid59/Real-Time-Tensorflow-Model-Layer-Visualization/blob/master/visualization_output.png)

## Quick Start Guide
Install the necessary dependencies.
```
sudo apt-get install libopencv-dev python-opencv
sudo pip install numpy scipy
```

