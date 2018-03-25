import cv2
import pyautogui
import numpy as np
from numpy import linalg


cap = cv2.VideoCapture(0)

sensity = 15
green_lwr = np.array([60 - sensity, 100, 50], np.uint8)
green_upr = np.array([60 + sensity, 255, 250], np.uint8)
red_lwr = np.array([60 - sensity, 100, 50], np.uint8)
red_upr = np.array([60 + sensity, 255, 250], np.uint8)


center_x, center_y = 0,0
mouse_x,mouse_y = center_x,center_y

font= cv2.FONT_HERSHEY_SIMPLEX

gestures = np.load('models/models.npy')
labels = np.load('models/model_labels.npy')


while (True):

    ret, img = cap.read()
    #img = cv2.flip(img, 1)

    
    #cv2.imshow('frame',gray)
    #cv2.rectangle(img,(300,300),(0,0),(0,255,0),0)
    

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    thresh = cv2.inRange(hsv, green_lwr, green_upr)

    ret, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    max_bound = 0

    if (contours):
        for i in contours:
            area = cv2.contourArea(i)

            # search for largest concentration of color
            if (area > max_bound):
                max_bound = area
                contour = i

        x,y,w,h = cv2.boundingRect(contour)
        x -= 100
        y -= 250
        w = x + 340
        h = y + 340
        if (x < 0): x = 0
        if (y < 0): y = 0
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        roi = img[y:h, x:w]
        #cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (35, 35), 0)
        _, edges = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        l = len(contours) 
        area = np.zeros(l)   
        for i in range(0, l):
            area[i] = cv2.contourArea(contours[i])
        
        index = area.argmax()
        #hand = contour
        hand = contours[index]
        #x,y,w,h = cv2.boundingRect(hand)
        
        temp = np.zeros(roi.shape, np.uint8)


        m = cv2.moments(hand)

        #cv2.drawContours(temp, [hand], -1, (0, 255, 0), -1)
        cv2.drawContours(roi, [hand], -1, (0, 255, 0), -1)


        measures = []
        for g in gestures:
            m = cv2.matchShapes(hand, g, 1, 0.0)
            measures.append(m)

        z = measures.index(min(measures))
        result = labels[z]

        cv2.putText(img,result, (600, 20), font, 0.7, (255, 255 ,255), 2, cv2.LINE_AA)


        if (result > "c"):
            print("click")
            #pyautogui.click()


        center_x = int((x + x + w) / 2)
        center_y = int((y + y + h) / 2)
        
        d_x = abs(mouse_x - center_x)
        d_y = abs(mouse_y - center_y)

        # stabalize 
        if (d_x > 20 or d_y > 20):
            mouse_x = center_x
            mouse_y = center_y

        pyautogui.moveTo(mouse_x, mouse_y)

        #cv2.circle(img, (center_x, center_y), 2, (0, 0, 255), 2)
        
    cv2.imshow('Astral', img)

    k = cv2.waitKey(10)
    if (k == 27):
        break
      

