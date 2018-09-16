import numpy as np
import sys
import os
import facenet
import tensorflow as tf
import configure
import cv2
import time


def load_model():
	graph = tf.Graph()
	sess = tf.Session(graph = graph)
	with graph.as_default():
		with sess.as_default():
			facenet.load_model("model")
	return graph, sess

def load_input(graph, sess):
	images_placeholder = graph.get_tensor_by_name("input:0")
	embeddings = graph.get_tensor_by_name("embeddings:0")
	phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
	return images_placeholder, embeddings, phase_train_placeholder

def embedding_path(paths, graph, sess, images_placeholder, embeddings, phase_train_placeholder):
	with graph.as_default():
		with sess.as_default():
			images =  facenet.load_data_paths(paths, False, False, configure.SIZEMODEL)
			feed_dict = {images_placeholder: images, phase_train_placeholder: False}
			emb_array = sess.run(embeddings, feed_dict = feed_dict)
	return emb_array

def embedding_image(images_list, graph, sess, images_placeholder,embeddings, phase_train_placeholder):
	with graph.as_default():
		with sess.as_default():
			images = facenet.load_data(images_list, False, False, configure.SIZEMODEL)
			feed_dict = {images_placeholder:images, phase_train_placeholder:False}
			emb_array = sess.run(embeddings, feed_dict = feed_dict)
	return emb_array

def distance_emb(emb_array1, emb_array2):
	return np.sum(np.square(np.subtract(emb_array1,emb_array2)))

# images = cv2.imread("hung.jpg")
# arr = []
# arr.append(images)
# arr.append(images)
# graph, sess = load_model()
# images_placeholder, embeddings, phase_train_placeholder = load_input(graph, sess)

# emb = embedding_image(arr, graph, sess, images_placeholder, embeddings, phase_train_placeholder)
# print (emb)