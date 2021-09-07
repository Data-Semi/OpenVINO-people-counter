"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


#import os
import sys
import time
import socket
import json
import cv2
import tensorflow as tf

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
TOPIC = "people_counter_python"
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
DEBUG_MODE = True

last_count = 0
total_count = 0
start_time = 0
current_count = 0
continues_threshold = 10
continues_frames_buff = 0
bef_count = 0
aft_count = 0
start_time = time.time()

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
                        #e.g. ./models/person-detection-0200.xml
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    
    # Note - CPU extensions are moved to plugin since OpenVINO release 2020.1. 
    # The extensions are loaded automatically while     
    # loading the CPU plugin, hence 'add_extension' need not be used.
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                             default=None,
                             help="MKLDNN (CPU)-targeted custom layers."
                                  "Absolute path to a shared library with the"
                                  "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    client.subscribe(TOPIC)
    return client
    

def infer_on_stream(args, client, single_image_mode=False):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()

    ### TODO: Load the model through `infer_network` ###
    model = args.model
    if model.endswith('.xml') :
        infer_network.load_model(model, args.device,args.cpu_extension)
        net_input_shape = infer_network.get_input_shape() # e.g.shape: 1, 3, 256, 256 in the format B, C, H, W
        #print(net_input_shape)
    ### TODO: Handle the input stream ###
    item = args.input
    if item.isdigit():
        input_stream = int(item)
    elif item.endswith('.jpg') or item.endswith('.bmp') :
        single_image_mode = True
        input_stream = item
    else:
        input_stream = item
    # Get and open video capture
    cap = cv2.VideoCapture(input_stream)
    if input_stream:
        cap.open(input_stream)
    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")

    # Set Probability threshold for detections
    global prob_threshold, initial_width, initial_height
    prob_threshold = args.prob_threshold
    # Grab the shape of the input 
    initial_width = int(cap.get(3))
    initial_height = int(cap.get(4))
    ### TODO: Loop until stream is over ###
    # Process frames until the video ends, or process is exited
    # to calculate average inference time
    det_time_sum = 0
    det_time_count = 0
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(10)
        ### TODO: Pre-process the image as needed ###
        # Pre-process the frame
        if model.endswith('.xml') :
            #Segmentation fault (core dumped), this is because the original frame do not have 600x600
            p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)
            ### TODO: Start asynchronous inference for specified request ###
            # Start asynchronous inference for specified request.
            inf_start = time.time()    
            # Perform inference on the frame
            infer_network.exec_net(p_frame)
            ### TODO: Wait for the result ###
            # Get the output of inference      
            if infer_network.wait() == 0:
                global current_count
                det_time = time.time() - inf_start
                det_time_sum = det_time_sum + det_time
                det_time_count = det_time_count + 1
                ### TODO: Get the results of the inference request ###
                result = infer_network.get_output()
                ### TODO: Extract any desired stats from the results ###
                frame, current_count = get_ppl_frame_count(frame, result)
                inf_time_message = "Inference time: {:.3f}ms"\
                                   .format(det_time * 1000)
                cv2.putText(frame, inf_time_message, (15, 15),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
                ### TODO: Calculate and send relevant information on ###
                ### current_count, total_count and duration to the MQTT server ###
                ### Topic "person": keys of "count" and "total" ###
                ### Topic "person/duration": key of "duration" ###

                client = count_total_and_duration(client)

        else:
            #without conversion, use pretrained TF model as is. eg. *.pb files
            inf_start = time.time()        
            ### TODO: Extract any desired stats from the results ###
                    # Read the graph.
            with tf.gfile.FastGFile(model, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            frame, current_count = infer_from_NON_conv_model(graph_def, frame)
            det_time = time.time() - inf_start
            det_time_sum = det_time_sum + det_time
            det_time_count = det_time_count + 1
            inf_time_message = "Inference time: {:.3f}ms"\
            .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client = count_total_and_duration(client)                
        if key_pressed == 27:
            break
            
        if DEBUG_MODE == False:
        ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode == True:
            cv2.imwrite('output_image.jpg', frame)

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    #Disconnect from MQTT
    client.disconnect()
    infer_network.clean()
    if DEBUG_MODE == True:
        avg_inf_time = det_time_sum / det_time_count
        print("Average inference time is:{:.3f}ms".format(avg_inf_time * 1000))
        

def count_total_and_duration(client):   
    global last_count, total_count, start_time
    global current_count, continues_threshold, continues_frames_buff
    global continues_frames_buff, bef_count, aft_count
    # To avoid noise, just take stable count as before and after
    if current_count == last_count:
        continues_frames_buff = continues_frames_buff + 1
        #the counted number continued # of frame, save bef continued and aft continued
        if continues_frames_buff > continues_threshold:
            bef_count = aft_count
            aft_count = current_count
    else:
        continues_frames_buff = 0

        #object detection count was stable
        #could start count a people as detected
#    print(aft_count, bef_count, continues_frames_buff, continues_threshold)
    if aft_count > bef_count and continues_frames_buff > continues_threshold:
            #newly came in a people
        start_time = time.time()
        total_count = total_count + aft_count - bef_count
        client.publish("person", json.dumps({"total": total_count}))
        # Person duration in the video is calculate
        # this current frame is right after #s of continues 0 detection, can calculate
    if aft_count < bef_count and continues_frames_buff > continues_threshold:
        duration = int(time.time() - start_time)
        # Publish messages to the MQTT server
        client.publish("person/duration",
                                   json.dumps({"duration": duration}))

    client.publish("person", json.dumps({"count": current_count}))
    last_count = current_count
    client.publish("current_count", json.dumps({"current_count": current_count}))
    
    return client
    
def get_ppl_frame_count(frame, result):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_width)
            ymin = int(obj[4] * initial_height)
            xmax = int(obj[5] * initial_width)
            ymax = int(obj[6] * initial_height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3) # color =RGB, Thickness =3
            current_count = current_count + 1

    return frame, current_count

def infer_from_NON_conv_model(graph_def, frame):

    with tf.compat.v1.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        #img = cv2.imread('example.jpg')
        img = frame
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv2.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        current_count = 0
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > prob_threshold:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                current_count = current_count + 1

    return img, num_detections

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Flag for the input image
    single_image_mode = False
    
    # Perform inference on the input stream
    infer_on_stream(args, client, single_image_mode)


if __name__ == '__main__':
    main()
    exit(0)
