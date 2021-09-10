# Project Write-Up
This is the first project in Udacity Intel® Edge AI for IoT Developers Nanodegree Program.
## Explaining Custom Layers
Custom layers are neural network model layers that are not natively supported by a given model framework.

The process behind converting custom layers involves two necessary custom layer extensions: Custom Layer Extractor and Custom Layer Operation.    
Custom Layer Extractor is responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged. In this project, it has been done by the function `load_model` in file `inference.py`.
Custom Layer Operation is responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters.   

Some of the potential reasons for handling custom layers are:  

1.When a layer isn’t supported by the Model Optimizer ,Model Optimizer does not know about the custom layers so it needs to taken care of and also need to handle for unsupported layers at the time of inference.  

2.Allow model optimizer to convert specific model to Intermediate Representation.  


## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were to compare the accuracy, average inference time(latency), CPU usage, and memory usage.

The difference between model accuracy pre- and post-conversion was small.  

The size of the model pre- and post-conversion was different, post-conversion has smaller size. For example, with the model `ssdlite_mobilenet_v2_coco_2018_05_09`, pre-conversion size is  , post-conversion size is 

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]
Models zoo link:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models

Documentation link to find a suitable configuration files:
https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [ssd_mobilenet_v2_coco_2018_03_29]
  - [Model Source] http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments.
  ```
  cd /home/workspace/ssd_mobilenet_v2_coco_2018_03_29  
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```

 - Run the main.py with following argument with the converted IR:  
   ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    ```
- I have got below result.  
    Average inference time is:69.115ms
    The accuracy is:0.921
    CPU: 86% (monitored by `top` command)
    Memory: 11% (monitored by `top` command)
    from the result, we can say the model was good for the app.
  - I tried to improve the model for the app by change the confidence threshold to 0.3
  - I found when I try to use `frozen_inference_graph.pb` fileinference directly from pre-conversion model, I need average 6second to inference a frame, it was impossible to inference a video because of the speed was too slow. 
  
- Model 2: [ssd_inception_v2_coco_2018_01_28]
  - [Model Source] http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments...
    ```
  cd /home/workspace/ssd_inception_v2_coco_2018_01_28
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json  
    ```

- Run the main.py with following argument with the converted IR:  
    ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm  
    ```  
- I have got below result.  
    Average inference time is:155.145ms
    The accuracy is:0.852
    CPU: 93% (monitored by `top` command)
    Memory: 13.3% (monitored by `top` command)
    from the result, we can say the model was good for the app.
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 3: [ssdlite_mobilenet_v2_coco_2018_05_09]
  - [Model Source] http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments...
  cd /home/workspace/ssdlite_mobilenet_v2_coco_2018_05_09
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

- Run the main.py with following argument with the converted IR:  
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm  
  
- I have got below result.  
    Average inference time is:31.102ms
    The accuracy is:0.859
    CPU: 76.4% (monitored by `top` command)
    Memory: 8.5% (monitored by `top` command)
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 4: [person-detection-retail-0013]
    Average inference time is:47.368ms
    The accuracy is:0.989
    CPU: 82% (monitored by `top` command)
    Memory: 7.5% (monitored by `top` command)

- Model 5: [ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18]
    http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz




  - The model `faster_rcnn_inception_v2_coco_2018_01_28` was insufficient for the app because I have get`segmentation fault (core dumped)`error, failed to load the model.
As same as the model `mask_rcnn_inception_v2_coco`, I can not get load model correctly. After load model, the model input size is only [1,3] instead of [1,3,H,W].

  ### Notes for converting models
  My main directory in Udacity's workspace was:  
  /home/workspace  

  1. Use command to see the version of tensorflow have been installed: 
  pip freeze | grep tensor or, python -m pip freeze | grep tensor
  The tensorflow version in my workspace was 1.14.0, therefore, I use below modelzoo links.  
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models  
  2. Instal prerequisites for TF.
    cd /opt/intel/openvino/deployment_tools/model_optimizer# cd install_prerequisites/
    source install_prerequisites.sh

  3. To download a model, I used command as below:
  wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz


  4. To convert a model to IR:
  refering to https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html  
  command:
  cd ssd_mobilenet_v2_coco_2018_03_29
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

  I have below output in the terminal by the conversion:
  [ SUCCESS ] Generated IR model.
  [ SUCCESS ] XML file: /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/./frozen_inference_graph.xml
  [ SUCCESS ] BIN file: /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/./frozen_inference_graph.bin
  [ SUCCESS ] Total execution time: 54.13 seconds.

  5. Feed the IR to my main.py
  cd /home/workspace/  
  source setup.sh  
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

  6. Get accuracy data etc.

  ### Notes for use TF models without converting to IR:
  https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

  I use below command line to inference directly from pre-conversion model.
  In this model it was using around average 6second to inference a frame, it was impossible to inference a video because of the speed was too slow.  

  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm



    Average inference time is:31.102ms
    The accuracy is:0.859
    CPU: 99.3% (monitored by `top` command)
    Memory: 68.1% (monitored by `top` command)






  

