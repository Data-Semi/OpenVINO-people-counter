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
import math

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
    #     parser.add_argument("-l", "--cpu_extension", required=False, type=str,
    #                         default=None,
    #                         help="MKLDNN (CPU)-targeted custom layers."
    #                              "Absolute path to a shared library with the"
    #                              "kernels impl.")
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
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3) # color =BGR, Thickness =3
            current_count = current_count + 1
    return frame, current_count

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
    infer_network.load_model(model, args.device)
    net_input_shape = infer_network.get_input_shape() # e.g.shape: 1, 3, 256, 256 in the format B, C, H, W
    #print("net_input_shape:", net_input_shape)
    ### TODO: Handle the input stream ###
    item = args.input
    if item == 'CAM':
        input_stream = -1
    else:
    #print("item is", item)
        if [item.endswith('.jpg') or item.endswith('.bmp')] :
            single_image_mode = True
            input_stream = item
        else:
            input_stream = item
    # Get and open video capture
    cap = cv2.VideoCapture(input_stream)
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
    last_count = 0
    total_count = 0
    start_time = 0
    current_count = 0
    # add previous counter to make sure not failed to detect people in previous frame
    last_last_count = 0
    last_last_last_count = 0
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(10)
        ### TODO: Pre-process the image as needed ###
        # Pre-process the frame
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
            det_time = time.time() - inf_start
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
            # When new person enters the video
            if current_count > last_count and last_count == last_last_count and last_last_count == last_last_last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            # Person duration in the video is calculated
            if current_count < last_count and last_count == last_last_count and last_last_count == last_last_last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",
                            json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_last_last_count = last_last_count
            last_last_count = last_count
            last_count = current_count
            client.publish("current_count", json.dumps({"current_count": current_count}))
            # Break if escape key pressed
        if key_pressed == 27:
            break
        #print("start-------sending----------------")
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        single_image_mode = True
        #print("-----------single_image-------------------------")
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode == True:
            cv2.imwrite('output_image.jpg', frame)

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    #Disconnect from MQTT
    client.disconnect()
    infer_network.clean()

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
