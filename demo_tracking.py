import cv2
import numpy as np 
import time

class Tracking_Face:
	def __init__(self, name, bounding_box, last_time):
		self.name = name
		self.bounding_box = bounding_box
		self.last_time = last_time
	def update_all(self, name, bounding_box, last_time):
		self.name = name
		self.bounding_box = bounding_box
		self.last_time = last_time
	def update_bounding_box(self, bounding_box):
		self.bounding_box = bounding_box

mutitracker = []

def update_mutitracker(face, predection):
	global mutitracker
	for idex, bounding_box in enumerate(face):
		identity = predection
		finder_tracker = has_tracker_control(bounding_box)
		if finder_tracker == -1:
			mutitracker.append(Tracking_Face(identity, bounding_box, time.time()))
			# print ("Create a new tracker")
		else :
			# print ("found tracker")
			tracker = mutitracker[finder_tracker]
			tracker.update_bounding_box(bounding_box)
			# print ("last time: ", time.time() - tracker.last_time)
			# if predection != "Unknow" or predection != "UNKNOW":
			# 	tracker.last_time = time.time()

	for idex, tracker in enumerate(mutitracker):
		check = -1
		for bounding_box in face:
			if bounding_box[0] == tracker.bounding_box[0] and bounding_box[1]==tracker.bounding_box[1]:
				if bounding_box[2] == tracker.bounding_box[2] and bounding_box[2] == tracker.bounding_box[2]:
					check = 1
		if check == -1:
			del mutitracker[idex]

def has_tracker_control(bounding_box):
	pos = -1
	max_area = -1
	global mutitracker
	for idex, tracker in enumerate(mutitracker):
		overlap = percent_intersection(tracker.bounding_box, bounding_box)
		if max_area < overlap:
			max_area = overlap
			pos = idex
	if max_area >= 20:
		return pos
	return -1

def percent_intersection(bounding_box1, bounding_box2):
	left = max(bounding_box1[0], bounding_box2[0])
	right = min(bounding_box1[2]+bounding_box1[0], bounding_box2[2]+bounding_box2[0])
	bottom = max(bounding_box1[1], bounding_box1[1])
	top = min(bounding_box1[1]+bounding_box1[3], bounding_box2[1]+bounding_box2[3])
	if left < right and bottom < top:
		s1 = bounding_box1[2]*bounding_box1[3]
		s2 = bounding_box2[2]*bounding_box2[3]
		intersecting_area = (right - left)* (top - bottom)
		return round((intersecting_area/(s1+s2 - intersecting_area)*100))
	else:
		return 0