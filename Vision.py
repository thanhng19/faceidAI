import tensorflow as tf 
import demo_tracking as tracking 
import model_detect as detect 
import model_embedding
import os
import sys
import facenet
import numpy as np 
import INFO
import _thread
import cv2
import time
import Model_Data as data
import pickle
from sklearn.neighbors import KNeighborsClassifier

graph, sess = model_embedding.load_model()
images_placeholder, embeddings, phase_train_placeholder = model_embedding.load_input(graph, sess) 

def load_source(list_img):
	emb_source = model_embedding.embedding_image(list_img, graph, sess, images_placeholder, embeddings, phase_train_placeholder) 
	return emb_source

def load_data_train(director):
	if os.path.isfile(director):
		data.customer = pickle.load(open(director, "rb"))


load_data_train("data_train")
print (len(data.customer))

kNN = KNeighborsClassifier(n_neighbors = 2, algorithm = "ball_tree", weights = "distance") 

def update_kNN():
	arr = []
	check_data = 0
	y = []
	for idx, customer in enumerate(data.customer):
		if idx == 0:
			arr = customer.embedding
		else :
			if idx < len(data.customer):
				check_data = 1
				arr = tf.concat([arr, customer.embedding], axis = 0)
		for emb in customer.embedding:
			y.append(idx)	
	if check_data == 1:
		x = tf.Session().run(arr)		
	else :
		x = arr
	kNN.fit(x,y)
update_kNN()

detect_minsize = 40	
detect_threshold = [ 0.6, 0.7, 0.7 ]
detect_factor = 0.709
pnet, rnet, onet = detect.load_data()

frame = None
stop = True
run = False
dem = 0
name_input = ""
run_detect = 0
run_emb = 0
list_img = []
arr_img_count = []
identify_count = []

cap = cv2.VideoCapture(0)

def run_video():
	global frame, stop, run_detect
	while stop:
		st = time.time()
		_, images = cap.read()
		run_detect = 1
		frame = images
		# print ("FPS", 1.0/(time.time() - st))

def run_detect_face():
	global frame, stop, run_detect, dem
	global list_img, run, identify_count, arr_img_count
	while stop:
		if run_detect == 1:
			faces = detect.detect_face_model(frame, detect_minsize, pnet, rnet, onet, detect_threshold, detect_factor)
			if run == False:
				st = time.time()
				run_detect = 2
				run_emb = 1
				for x,y,w,h in faces:
					cv2.rectangle(frame,(max(0,x),max(0,y)),(x+w,y+h),(255,0,0),2)
				tracking.update_mutitracker(faces, "UNKNOW")
				arr_images = []
				font = cv2.FONT_HERSHEY_SIMPLEX
				for idx ,tracker in enumerate(tracking.mutitracker):
					x = tracker.bounding_box[0]
					y = tracker.bounding_box[1]
					w = tracker.bounding_box[2]
					h = tracker.bounding_box[3]
					image = frame[max(0,y):y+h, max(0,x):x+w]
					thumbnail = cv2.resize(image, (160, 160), interpolation = cv2.INTER_AREA)
					if tracker.name == "UNKNOW":
						if time.time() - tracker.last_time > 1:
							cv2.rectangle(frame,(max(0,x),max(0,y)),(x+w,y+h),(0,0,255),2)
							cv2.putText(frame, tracker.name, (x-10, y-10) , font, 1,(0,0,255),1,cv2.LINE_AA)
						else :
							arr_images.append(thumbnail)
							identify_count.append(idx)
							arr_img_count.append(thumbnail)
							cv2.rectangle(frame,(max(0,x),max(0,y)),(x+w,y+h),(255,0,0),2)
							cv2.putText(frame, "Loading", (x-10, y-10) , font, 1,(255,0,0),1,cv2.LINE_AA)
					if tracker.name!= "UNKNOW":
						cv2.rectangle(frame,(max(0,x),max(0,y)),(x+w,y+h),(0,255,0),2)
						cv2.putText(frame, tracker.name, (x-10, y-10) , font, 1,(0,255,0),1,cv2.LINE_AA)
				cv2.imshow("frame", frame)
				arr_img = np.copy(arr_images)
				if len (arr_img) > 0 and len(data.customer) > 0:
					counting_embedding()
					print ("FPS: ", 1.0/(time.time() - st))
			k = cv2.waitKey(2) & 0xff
			if k == 27:
				stop = False
				break
			if k == ord('s'):
				run = True
				dem = 0
			if run == True:
				if len(faces) > 0 and len(faces) <=1:
					if dem == 4:
						print ("nhap ten cho class: ")
						name_input = input()
						run = False
					if dem < 4:
						for x,y,w,h in faces:
							image = frame[y:y+h, x:x+w]
							thumbnail = cv2.resize(image, (160, 160), interpolation = cv2.INTER_AREA)
							list_img.append(thumbnail)
							emb_source = load_source(list_img)
						if dem == 3:
							print ("size list images: ", len(list_img))
							data.customer.append(data.Customer(name_input, emb_source))
							pickle.dump(data.customer, open("data_train", "wb"))
							print (len(list_img))
							update_kNN()
							list_img = []
					dem += 1


def counting_embedding():
	global frame
	arr_img = []
	identify = []
	i = 0
	for idx, tracker in enumerate(tracking.mutitracker):
		if tracker.name == "UNKNOW":
			x = tracker.bounding_box[0]
			y = tracker.bounding_box[1]
			w = tracker.bounding_box[2]
			h = tracker.bounding_box[3]
			images = frame[max(0,y):y+h, max(0,x):x+w]
			thumbnail = cv2.resize(images, (160, 160), interpolation = cv2.INTER_AREA)
			arr_img.append(thumbnail)
			identify.append(idx)
			i += 1
	emb_data = []
	if len(arr_img) > 0:
		emb_data = model_embedding.embedding_image(arr_img, graph, sess, images_placeholder, embeddings, phase_train_placeholder)
	if len(emb_data) > 0:
		i = 0
		prediction = np.ndarray.tolist(kNN.predict(emb_data))
		print ("prediction: ", prediction)
		for emb_array1 in emb_data:	
			min = 10.0
			name = "UNKNOW"
			for emb_array2 in data.customer[prediction[i]].embedding:
				distance = model_embedding.distance_emb(emb_array1, emb_array2)
				if min > distance:
					min = distance
					name = data.customer[prediction[i]].name

			if min >= 0.6:
				name = "UNKNOW"
			tracking.mutitracker[identify[i]].name = name
			i += 1

try :
	_thread.start_new_thread(run_video, ())
	_thread.start_new_thread(run_detect_face, ())
except:
	print ("erro")
while stop:
	time.sleep(0.1)
