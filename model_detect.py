import detect_face
import configure
import model_embedding
import numpy as np 
import tensorflow as tf 
import sys
import os
import cv2
import time

def load_data():
	sess = tf.Session()
	pnet, rnet, onet = detect_face.create_mtcnn(sess, configure.MODEL_DETECT)
	return pnet, rnet, onet

def detect_face_model(frame, detect_minsize,pnet, rnet, onet, detect_threshold, detect_factor):
	list_bouding_box = []
	# if frame == None:
	# 	return list_bouding_box	
	total_box, point = detect_face.detect_face(frame, detect_minsize, pnet, rnet, onet, detect_threshold, detect_factor)
	for box in total_box:
		x = (int)(box[0])
		y = (int)(box[1])
		w = (int)(box[2] - box[0])
		h = (int)(box[3] - box[1])
		list_bouding_box.append((x,y,w,h))
	return list_bouding_box

# detect_minsize = 40	
# detect_threshold = [ 0.6, 0.7, 0.7 ]
# detect_factor = 0.709
# pnet, rnet, onet = load_data()
# detect_face_model(None, detect_minsize,pnet, rnet, onet, detect_threshold, detect_factor)