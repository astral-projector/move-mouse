import json
import cv2
import pyautogui
import numpy as np
from numpy import linalg

def set_color(color):
    sensity = 15

    if (color == "r"):
        lower = np.array([170 - sensity, 70, 50], np.uint8)
        upper = np.array([180 + sensity, 255, 255], np.uint8)

    else:   #default green
        lower = np.array([60 - sensity, 100, 50], np.uint8)
        upper = np.array([60 + sensity, 255, 250], np.uint8)

    return (upper,lower)

def move_mouse(x,y,w,h,mouse):
	mouse_x = x
	mouse_y = y
	center_x = int((x + x + w) / 2)
	center_y = int((y + y + h) / 2)

	d_x = abs(mouse_x - center_x)
	d_y = abs(mouse_y - center_y)

	if (d_x > 20 or d_y > 20):
		mouse_x = center_x
		mouse_y = center_y

	if (mouse):
		pyautogui.moveTo(mouse_x, mouse_y)


def recognize_shape(img, m, hand_contour):
	models = np.load('models/models.npy')
	labels = np.load('models/model_labels.npy')
	font = cv2.FONT_HERSHEY_SIMPLEX

	measures = []
	for g in models:
	    m = cv2.matchShapes(hand_contour, g, 1, 0.0)
	    measures.append(m)

	z = measures.index(min(measures))
	result = labels[z]

	cv2.putText(img,result, (600, 20), font, 0.7, (255, 255 ,255), 2, cv2.LINE_AA)

	return (result)


def shape_hand(img, roi):
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (35, 35), 0)
	_, edges = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	_, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	l = len(contours)
	area = np.zeros(l)
	max_bound = 0
	if (contours):
		for i in contours:
			area = cv2.contourArea(i)

			if (area > max_bound):
				max_bound = area
				contour = i

		# for i in range(0, l):
		# 	area[i] = cv2.contourArea(contours[i])

		# index = area.argmax()
		# hand = contours[index]

		#temp = np.zeros(roi.shape, np.uint8)

		m = cv2.moments(contour)
		cv2.drawContours(roi, [contour], -1, (0, 255, 0), -1)

		return (m, contour)
	return 


def find_hand(img, contour, mouse):
	x,y,w,h = cv2.boundingRect(contour)

	x -= 100
	y -= 250
	w = x + 340
	h = y + 340
	if (x < 0): x = 0
	if (y < 0): y = 0

	move_mouse(x,y,w,h,mouse)
	cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
	roi = img[y:h, x:w]

	return (roi)


def find_color(img, color):

	upper, lower = set_color(color)

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	thresh = cv2.inRange(hsv, lower, upper)
	ret, contours, hierarchy = cv2.findContours(thresh, 1, 2)

	l = len(contours)
	area = np.zeros(l)
	if (contours):
		for i in range(0, l):
			area[i] = cv2.contourArea(contours[i])

		index = area.argmax()
		hand = contours[index]

		return (hand)



def start_recording(mode, window_name, color, mouse, click):
	center_x, center_y = 0,0
	mouse_x, mouse_y = center_x, center_y

	video = cv2.VideoCapture(0)

	if (video.isOpened()):
		stream, img = video.read()
	else:
		print("no webcam found :(")
		return

	if (mode == "dev"):
		img = cv2.flip(img, 1)

	while (stream):
		contour = find_color(img, color)
		roi = find_hand(img, contour, mouse)
		m,hand_contour = shape_hand(img, roi)
		result = recognize_shape(img, m, hand_contour)

		if (result > "c" and mouse):
			print("click")

		k = cv2.waitKey(10)
		if k == 27:  # exit on esc
			break



if __name__ == '__main__':

	with open('config.json', 'r') as f:
	    config = json.load(f)

	name = config['SETTINGS']['name'] 
	mode = config['SETTINGS']['mode'] 
	color = config['SETTINGS']['color'] 
	mouse = config['SETTINGS']['mouse'] 
	click = config['SETTINGS']['click'] 

	start_recording(mode, name, color, mouse, click)



