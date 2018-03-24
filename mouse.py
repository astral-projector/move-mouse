import cv2
import pyautogui
import numpy as np

video = cv2.VideoCapture(0)

sensity = 15
green_lwr = np.array([60 - sensity, 100, 50], np.uint8)
green_upr = np.array([60 + sensity, 255, 250], np.uint8)
center_x, center_y = 200,200
mouse_x,mouse_y = center_x,center_y


while (video.isOpened()):

    ret, img = video.read()
    img = cv2.flip(img, 1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    thresh = cv2.inRange(hsv, green_lwr, green_upr)

    ret, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    max_bound = 0

    if contours:
        for i in contours:
            area = cv2.contourArea(i)

            if area > max_bound:
                max_bound = area
                contour = i

        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        

        center_x = int((x + x + w) / 2)
        center_y = int((y + y + h) / 2)
        
        d_x = abs(mouse_x - center_x)
        d_y = abs(mouse_y - center_y)

        if (d_x > 20 or d_y > 20):
            mouse_x = center_x
            mouse_y = center_y

        pyautogui.moveTo(mouse_x, mouse_y)

        cv2.circle(img, (center_x, center_y), 2, (0, 0, 255), 2)

        cv2.imshow('Astral', img)

        
       


