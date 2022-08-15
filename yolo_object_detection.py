#!/usr/bin/env python3
import cv2
import rospy
import numpy as np
import glob
import time
import imutils
import sys

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.4

class ObjectDetection:
	def __init__(self):
		print("__init__ called.")
		self.bridge = CvBridge()
		
		# Set to ROS topic
		img_topic = "/image_jpeg/compressed"
		self.image_sub = rospy.Subscriber(img_topic, CompressedImage, self.callback)
		
		# Path to the YOLOv4 
		yolo_weights = glob.glob("./yolov4-tiny.weights")[0]
		yolo_cfg = glob.glob("./yolov4-tiny.cfg")[0]
		yolo_labels = glob.glob("./labels.txt")[0]
		
		# Read trained label
		self.class_names = list()
		with open(yolo_labels,"r") as f:
			self.class_names = [cname.strip() for cname in f.readlines()]
		
		# Set to the Deep Nueral Network(Darknet)
		self.net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
		
		'''
		Use cuda(GPU) - Need opencv-contrib-python cuda build
		'''
		# self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		# self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		
		# Set to the network layer
		self.layer = self.net.getLayerNames()
		self.layer = [self.layer[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		
		# Set to Bbox color
		self.COLORS = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype="uint8")

	'''
	ROS Simulator compressed image msg 
	'''
	def callback(self, data):
		print("CB")
		try:
			print(data.header)
			print(sys.version)
			compressed_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
			# compressed_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as error_str:
			print(error_str)
		
		# YOLO Detection
		detect_image = detect(compressed_image, self.net, self.layer, self.COLORS, self.class_names)
		print("detect_image created")
		
		# Visualization
		cv2.imshow("", detect_image)
		print("imshow")
		cv2.waitKey(1)

'''
YOLOv4 Detect Function
'''
def detect(frame, net, layer, Bbox_COLORS, class_names):
	(H, W) = frame.shape[:2]
	
	# Blob objectification of input images for inference
	start_time = time.time()
	blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(layer)
	end_time = time.time()
	
	boxes = []
	classIds = []
	confidences = []
	
	for output in layerOutputs:
		for detection in output:
		    scores = detection[5:]
		    classID = np.argmax(scores)
		    confidence = scores[classID]
		    
		    if confidence > CONFIDENCE_THRESHOLD:
		        box = detection[0:4] * np.array([W, H, W, H])
		        (centerX, centerY, width, height) = box.astype("int")
		        
		        x = int(centerX - (width/2))
		        y = int(centerY - (height/2))

		        boxes.append([x, y, int(width), int(height)])
		        classIds.append(classID)
		        confidences.append(float(confidence))
		        
	# Non-Maximum Supression
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	
	# Print fps(frame per second) on frame
	fps_label = "FPS: %.2f" % (1 / (end_time - start_time))
	cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
	
	if len(idxs) > 0:
		for i in idxs.flatten():
		    (x, y) = (boxes[i][0], boxes[i][1])
		    (w, h) = (boxes[i][2], boxes[i][3])

		    color = [int(c) for c in Bbox_COLORS[classIds[i]]]
		    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
		    text = "{}: {:.4f}".format(class_names[classIds[i]], confidences[i])
		    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
	
	return frame

if __name__ == '__main__':
    obj_det_ = ObjectDetection()
    rospy.init_node('CompressedImages', anonymous=False)
    rospy.spin()
