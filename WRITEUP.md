# Deploying a People Counter App at the Edge
---

# Project Write-Up
This is the first project in Udacity Intel® Edge AI for IoT Developers Nanodegree Program.

---

## Explaining Custom Layers
Custom layers are neural network model layers that are not natively supported by a given model framework.   

The process behind converting custom layers involves two necessary custom layer extensions: Custom Layer Extractor and Custom Layer Operation.    

1. Custom Layer Extractor is responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged. In this project, it has been done by the function `load_model` in file `inference.py`.  
2. Custom Layer Operation is responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters.   

- Each device plugin in the Inference Engine includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension is implemented according to the target device. For example,Custom Layer CPU Extension is a compiled shared library (`.so` or `.dll` binary) needed by the CPU Plugin for executing the custom layer on the CPU.  

- The Model Extension Generator tool generates template source code files for each of the extensions needed by the Model Optimizer and the Inference Engine.  

- The script for this is available here-  `/opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py`  

```
optional arguments:
  -h, --help            show this help message and exit
  --mo-caffe-ext        generate a Model Optimizer Caffe* extractor
  --mo-mxnet-ext        generate a Model Optimizer MXNet* extractor
  --mo-tf-ext           generate a Model Optimizer TensorFlow* extractor
  --mo-op               generate a Model Optimizer operation
  --ie-cpu-ext          generate an Inference Engine CPU extension
  --ie-gpu-ext          generate an Inference Engine GPU extension
  --output_dir OUTPUT_DIR
                        set an output directory. If not specified, the current
                        directory is used by default.
```  

Some of the potential reasons for handling custom layers are:  
1. When a layer isn’t supported by the Model Optimizer ,Model Optimizer does not know about the custom layers so it needs to taken care of and also need to handle for unsupported layers at the time of inference.  

2. Allow model optimizer to convert specific model to Intermediate Representation.   

---
## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations were to compare the accuracy, average inference time(latency), CPU usage, and memory usage.  
Because of the plain TF models uses too many memory and CPU, I decided to only inference the first 80 frames of the video for the model `ssdlite_mobilenet_v2_coco_2018_05_09` just for reference. You can find `pb_early_stopper` variable in `main.py`to get the same performance.   

The table below is evident that keep small accuracy decreases but Latency (microseconds) and Memory (Mb) CPU(%) decreases very much in case of OpenVINO as compared to plain Tensorflow model.
I have used `top` command to monitor CPU and memory usase.
```
top -p 23512 -b -d 10 -n 30 > top_result.txt
```


| Model/Framework                                 | Latency (microseconds)            | Memory (%) | CPU(%) | Accuracy |
| ------------------------------------------------|:---------------------------------:| -------:| -------:| -------:|         
| ssdlite_mobilenet_v2_coco (plain TF)            | 9372                              | 92.5   | 91.9     | 0.963   |  
| ssdlite_mobilenet_v2_coco (Post-conversion)     | 7                                 | 3      | 180      | 0.859   |  
| ssd_mobilenet_v2_coco (Post-conversion)         | 12                                | 4      | 270      | 0.921   |  
| ssd_inception_v2_coco (Post-conversion)         | 25                                | 4      | 381      | 0.852   |
| person-detection-retail-0013 (OpenVINO)         | 9.8                               | 2.5    | 240      | 0.984   |   

---

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
1. Count people in a specific place, for example we can count how many people came to a shop in a day.  
2. Compare what kind of product is popular in customer by comparing of the duration of every customer stay in front the product.  
3. Monitor how long a helper spend time on a patients in a time period.  

Each of these use cases would be useful because we can get more accurate data with low man hour cost and also avoid human error and risk during COVID-19 era.

---

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model.   

The potential effects of each of these are as follows:
1. In case of poor lighting model's accuracy may fail considerably or even completely. However, this can be mitigated with good hardware that can process the images before passing it to the model.    
2. Low model accuracy will get poor detection information that difficult to do further calculation. An effective solution to this is to ignore some detection failing and result fluctuation. For example, you can find the variable `continues_threshold` in `main.py` that make calculation only wen inference result on frames are continuesly(more than the given threshold) have the same detection result.   
3. Distorted input from camera due to change in focal length and/or image size will affect the model because the model may fail to make sense of the input. An approach to solve this would be to pre-process the image to make the contrast better or augmented images before feed in to models.

---

## Model Research
Models zoo link:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models

Documentation link to find a suitable configuration files:
https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [ssdlite_mobilenet_v2_coco]
  - [Model Source] http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
  - Download and extract files.
  ```
  wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
  tar -xvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
  ```
  - I converted the model to an Intermediate Representation with the following arguments.
  ```
  cd ssdlite_mobilenet_v2_coco_2018_05_09
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - Run the main.py with following argument with the converted IR:
  ```  
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ./ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm  
  ``` 
  - The model was insufficient for the app because it is failing to detect in some frame.
  - I tried to improve the model for the app by change the threshold to 0.3 and add a variable `continues_threshold` to 10 in `main.py` that make calculation only wen inference result on frames are continuesly(more than the given threshold) have the same detection result. 

  - Test pre-conversion model  
  With the code referenced from [here](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API), I used below command line to inference directly from pre-conversion model.
  It was using around average 9second to inference a frame, I decided to only inference the first 80 frames of the video for the model `ssdlite_mobilenet_v2_coco_2018_05_09` just for reference. You can find `pb_early_stopper` variable in `main.py`to get the same performance.  
  ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ./ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  ``` 
---

- Model 2: [ssd_mobilenet_v2_coco]
  - [Model Source] http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - Download and extract files. Same command as Model 1 above.
  - I converted the model to an Intermediate Representation with the following arguments.
  ```
  cd ssd_mobilenet_v2_coco_2018_03_29  
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - Run the main.py with following argument with the converted IR:  
  ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ./ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  ```
  - from the result, we can say the model was good for the app.
  - I tried to improve the model for the app by change the threshold to 0.3 and add a variable `continues_threshold` to 10 in `main.py` that make calculation only wen inference result on frames are continuesly(more than the given threshold) have the same detection result.
  - I have tried to test pre-conversion model with this Model 2. The performance is similar to Model 1.

---

- Model 3: [ssd_inception_v2_coco_2018_01_28]
  - [Model Source] http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
  - Download and extract files. Same command as Model 1 above.
  - I converted the model to an Intermediate Representation with the following arguments.
  ```
  cd ssd_inception_v2_coco_2018_01_28
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json  
  ```
  - Run the main.py with following argument with the converted IR:  
  ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ./ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm  
  ```  
  - The model was insufficient for the app because it is failing to detect in some frame.
  - I tried to improve the model for the app by change the threshold to 0.3 and add a variable `continues_threshold` to 10 in `main.py` that make calculation only wen inference result on frames are continuesly(more than the given threshold) have the same detection result.

---

- Model 4: [person-detection-retail-0013]
  - Download from OpenVINO directly:
  ```
  cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
  python downloader.py --name person-detection-retail-0013 --precisions FP16 -o /mnt/c/pj/edgeai/OpenVINO-people-counter
  ```
  - Run the main.py with following argument with the converted IR:  
  ```
  cd /mnt/c/pj/edgeai/OpenVINO-people-counter
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ./intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm  
  ```  
  - from the result, we can say the model was good for the app.
  - I tried to improve the model for the app by change the threshold to 0.3 and add a variable `continues_threshold` to 10 in `main.py` that make calculation only wen inference result on frames are continuesly(more than the given threshold) have the same detection result.

---

- Other models
  - The model `faster_rcnn_inception_v2_coco_2018_01_28` was insufficient for the app because I have get`segmentation fault (core dumped)`error, failed to load the model. As same as the model `mask_rcnn_inception_v2_coco`, I can not get load model correctly. After load model, the model input size was only [1,3] instead of [1,3,H,W].

---
---
# Notes for converting models steps
  1. Use command to see the version of tensorflow have been installed: 
  pip freeze | grep tensor or, python -m pip freeze | grep tensor
  The tensorflow version in my workspace was 1.14.0, therefore, I use below modelzoo links.  
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models  

  2. Install prerequisites for TF.
  ```
    cd /opt/intel/openvino/deployment_tools/model_optimizer
    cd install_prerequisites/
    source install_prerequisites.sh
  ```

  3. To download a model, command as below:
  ```
  wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  ```

  4. To convert a model to IR:
  Refer to https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html   

  ```
  cd ssd_mobilenet_v2_coco_2018_03_29
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```

  5. Feed the IR to my `main.py`
  ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  ```

  6. Get accuracy data etc with `eval.py`

---
# Notes for setting up a local environment
## 1. OpenVINO installation guide:
  https://docs.openvinotoolkit.org/2019_R3/_docs_install_guides_installing_openvino_linux.html 

## 2. install wsl Ubuntu 16.04 and reboot
https://docs.microsoft.com/ja-jp/windows/wsl/install-win10
```
wsl --list --online
wsl --install -d Ubuntu-16.04
```
## 3. To make symbolic link that python command uses python 3.5
```
whereis python
sudo ln -s /usr/bin/python3.5 /usr/bin/python
```
## 4. Open Ubuntu 16.04 as administrator
```
cd /mnt/c/pj/edgeai/Openvino...
code .
```

## 5. Folow the set up guides from Udacity
https://github.com/udacity/nd131-openvino-fundamentals-project-starter/blob/master/windows-setup.md

---
## 6. Errors and solutions
---
### 6-1. Error: pip version 8.1.1
When perform command below, the error occurs.  
```
sudo ./install_prerequisites.sh
```
`python -m pip install --upgrade pip` is not soving this error.  

Solution is as below:  
```
wget https://bootstrap.pypa.io/pip/3.5/get-pip.py
python get-pip.py
```
Try again:  
```
cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
sudo ./install_prerequisites.sh
```
The symbolic link has been changed that python command usese python 2.7.  
Force to make symbolic link again.  
```
sudo ln -f -s /usr/bin/python3.5 /usr/bin/python
```
Continue OpenVINO's setup.  
```
cd /opt/intel/openvino/deployment_tools/demo
./demo_squeezenet_download_convert_run.sh
```
`pip install` modules needed when error occurs.  

---
### 6-2. Error: 'Graph' object has no attribute 'node'  
```
pip3 install networkx==2.3
```
Error again:  
[ ERROR ] Error loading xmlfile: /home/liang/openvino_models/ir/FP16//public/squeezenet1.1/squeezenet1.1.xml, File was not found at line: 1 pos: 0
Error on or near line 239; exiting with status 1
Solution:  
```
rm -r /home/liang/openvino_models/ir/FP16//public/squeezenet1.1
```
---
### 6-3. Error Gtk-WARNING **: cannot open display: :0  
When do the demo `./demo_security_barrier_camera.sh`, I get the error.  
With below solution the error did not disapeared, but I would like to skip it because it is not related to this project.  
```
export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
```
Error again.  
Gtk-WARNING **: cannot open display: 172.31.176.1:0.0

---

### 6-4 Error related to nodejs and npm
Followed Udacity's guide, after process command below install the nodejs again.  
```
sudo npm install npm -g 
rm -rf node_modules
npm cache clean
npm config set registry "http://registry.npmjs.org"
npm install
```
Remove before reinstall nodejs:  
```
sudo apt-get remove nodejs
```

Reinstall npm:  
```
cd webservice/server
npm update
npm install
```
---







  

