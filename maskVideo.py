import cv2
import numpy as np

cap = cv2.VideoCapture(0)

global biggest_contour
global next_biggest_contour
global biggest_rect
global next_biggest_rect

counter = 0
while (True):
    # Take each frame
    _, frame = cap.read()
    blur = cv2.medianBlur(frame, 5)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_green = np.array([0, 92, 30])
    upper_green = np.array([93, 255, 255])
    # hold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    biggest_contour = 0
    next_biggest_contour = 0
    biggest_rect = 0
    next_biggest_rect = 0

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(blur, blur, mask=mask)
    if len(contours) > 0:
        cnt = contours[len(contours) - 1]
        cv2.drawContours(res, [cnt], 0, (0, 255, 0), 3)
        # print("Area: " + str(cv2.contourArea(cnt)))
        # x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)

        biggest_contour = 0
        next_biggest_contour = 0
        biggest_rect = 0
        next_biggest_rect = 0

        x = 0
        y = 0
        w = 0
        h = 0

        xx = 0
        yy = 0
        ww = 0
        hh = 0
        for cont in contours:
            currentContour = cv2.contourArea(cont)
            if currentContour > biggest_contour:
                biggest_contour = currentContour
                x, y, w, h = cv2.boundingRect(cont)

            elif currentContour > next_biggest_contour:
                next_biggest_contour = currentContour
                xx, yy, ww, hh = cv2.boundingRect(cont)

        centerX = 0
        centerY = 0
        centerW = (xx + ww) - x
        centerH = (yy - hh) - y
        cv2.rectangle(res, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)
        cv2.rectangle(res, (x, y), (x + w, y + h), (255, 0, 0), 2)
        ##Euclidean Distance h = y2 - y1, w = x2 - x1
        boundingBox = 0

        blueBoundingBox = h * w
        redBoundingBox = hh * ww
        # if smaller contour is on the left
        if (xx < x):
            if (yy + hh) > (y + h):
                r = cv2.rectangle(res, (xx, yy), (x + w, yy + hh), (0, 255, 255), 2)
                boundingBox = hh * (x + w - ww)
                # midpoint formula
                # need to round centerxy to use in circle
                centerX = (xx + x + w) / 2
                centerY = (yy + yy + hh) / 2
                cv2.circle(res, (round(centerX), round(centerY)), 3, (0, 255, 0), -1)
            else:
                r = cv2.rectangle(res, (xx, yy), (x + w, y + h), (0, 255, 0), 2)
                boundingBox = (y + h - yy) * (x + w - xx)
                centerX = (xx + x + w) / 2
                centerY = (yy + y + h) / 2
                cv2.circle(res, (round(centerX), round(centerY)), 3, (0, 255, 0), -1)
        else:
            if (yy + hh) > (y + h):
                print("Going down")
                # TODO maybe instead of xx + ww, xx + yy
                r = cv2.rectangle(res, (x, y), (xx + ww, yy + hh), (0, 255, 255), 2)
                boundingBox = (yy + hh - y) * (xx + yy - y)
                centerX = (x + xx + ww) / 2
                centerY = (y + yy + hh) / 2
                cv2.circle(res, (round(centerX), round(centerY)), 3, (0, 255, 0), -1)
            else:
                r = cv2.rectangle(res, (x, y), (xx + ww, y + h), (0, 255, 0), 2)
                boundingBox = h * (xx + ww - x)
                centerX = (x + xx + ww) / 2
                centerY = (y + y + h) / 2
                cv2.circle(res, (round(centerX), round(centerY)), 3, (0, 255, 0), -1)
        print("BIggest Area " + str(biggest_contour))
        print("Next Biggest Area " + str(next_biggest_contour))
        print("area of bounding box " + str(boundingBox))
        print("Blue Bounding Box: " + str(blueBoundingBox))
        print("Red Bounding Box: " + str(redBoundingBox))
        print("Box Ratio = " + str((blueBoundingBox + redBoundingBox) / boundingBox))
        print("Midpoint is (" + str(centerX) + ", " + str(centerY) + ")")
        # print("Contour Ratio = " + str((biggest_contour + next_biggest_contour) / boundingBox))

        cv2.imshow('frame', frame)
        cv2.imshow('blur', blur)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        counter += 1

cv2.destroyAllWindows()
