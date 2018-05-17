# DIY Real Time Tensorflow Model Layer Visualization
This presents sample code to visualize some of the layers of a model loaded into tensorflow. We describe the process of choosing these layers and displaying them to screen.

## Overview
The image shows the real time visualization of the layers as the system is processing input from a webcam. We use the ssd_inception_v2_coco network from [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). The layers are hand picked and a random subset are plotted together.

![Sample Image Output during visualization of the layers](https://github.com/omarabid59/Real-Time-Tensorflow-Model-Layer-Visualization/blob/master/visualization_output.png)

## Quick Start Guide
Install the necessary dependencies, then follow the tutorial to [Install Tensorflow and the Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) for your OS.
```
sudo apt-get install libopencv-dev python-opencv
sudo pip install numpy scipy
```

Run the notebook **notebook name**.

## Specifying your own layers
To specify your own layers in the model to visualize is as follows:
1. Find the name of the Tensors we want to visualize, choosing only the ReLU layer map activations since these give the neural output when an input image is presented.
2. Set this as the input to our session.
3. Visualize the result.

#### Find the Tensor Names
The tensor names are required to tell the system which layers of the network to visualize. To get a list of the tensors, we can use the following code.
```
for op in graph.get_operations():
    if 'Relu' in op.name:
        print(op.name)
```
And the output looks something like this:
```
FeatureExtractor/InceptionV2/InceptionV2/Conv2d_1a_7x7/Relu6
FeatureExtractor/InceptionV2/InceptionV2/Conv2d_2b_1x1/Relu6
FeatureExtractor/InceptionV2/InceptionV2/Conv2d_2c_3x3/Relu6
...
```
Note that there will be a large amount of layers to choose from. Choosing one is a bit of a trial and error process. If you have a better idea of finding the layers we wish to visualize, please let me know!

#### Set as input to the session
Now, say we decide to pick the first layer to visualize. In the ```predict``` method, add the code below. Notice that we append the string with the ```:0``` at the end. Don't foget to do this or the code will not work! There's no limit to the number of layers we can visualize at a time.
```
layer = detection_graph.get_tensor_by_name(
          'FeatureExtractor/InceptionV2/InceptionV2/Conv2d_1a_7x7/Relu6:0')
```
Next, we need to tell our session to output the result of this process. We have two options. If we want the result of ONLY the visualization layers, we can run the following code:
```    
layer_1_out = sess.run([layer_1, feed_dict={image_tensor: image_np_expanded})
output_frame.visualization_data = [layer_1_out]
```
If we want to do the visualization in conjunction with outputing the classification and bounding box results of our SSD model, then we can run the following code:
```    
[layer_1_out, temp_box, temp_score, 
      temp_class,num_detections] = sess.run([layer_1,
          boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
output_frame.boxes = [temp_box, temp_score, temp_class,num_detections]
output_frame.visualization_data = [layer_1_out]
```

#### Visualize the result
To visualize the result, we can call the ```display_activations``` function as shown below.
```
display_activations(output_frame.visualization_data[0],num_features=6,padding=5,filter_size=(-1,-1))
```
To customize the visualization, there are several parameters we can play with.
1. ```num_features:``` Remember that our layer output - ```layer_1_out``` will be a tensor of the form (1,m_width,m_height, num_features). Thus, we can choose the number of features we want to visualize. To get the shape of a tensor, use ```output_frame.visualization_data[0].shape```.
2. ```padding:``` Spacing between each feature output.
3. ```filter_size:``` Size, in pixels of the feature size. If none is specified, uses the size of the first feature as the starting point.



