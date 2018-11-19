#!/usr/bin/env python3

import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -1)

#Calculated by hand
HFOV = 55.794542
VFOV = 50.92
#Calculated in program
H_FOCAL_LENGTH = 808
V_FOCAL_LENGTH = 707

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

change_res(640, 480)
# Masks the video based on a range of hsv colors
# Takes in a frame, returns a masked frame
def threshold_video(frame):

    #Gets the shape of video
    screenHeight, screenWidth, channels = frame.shape
    #Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    blur = cv2.medianBlur(frame, 5)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # define range of blue (my notebook) in HSV
    lower_color = np.array([0, 135, 53])
    upper_color = np.array([42, 218, 255])
    # hold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Shows the theshold image in a new window
    cv2.imshow('threshold', mask)
    # Returns the masked imageBlurs video to smooth out image

    return mask

#Finds the contours from the masked image and displays them on original stream
def findContours(cap, mask):
    #Finds contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Take each frame
    _, frame = cap.read()
    #Flips the frame so my right is the image's right (probably need to change this
    #frame = cv2.flip(frame, 1)
    #Gets the shape of video
    screenHeight, screenWidth, channels = frame.shape
    #Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    #Copies frame and stores it in image
    image = frame.copy()
    #Processes the contours, takes in (contours, output_image, (centerOfImage) #TODO finding largest
    if len(contours) != 0:
        processLargestContour(contours, image, centerX, centerY)
    #Shows the contours overlayed on the original video
    cv2.imshow("Contours", image)

#Draws and calculates properties of largest contour
def processLargestContour(contours, image, centerX, centerY):
    if len(contours) != 0:

        cnt = findBiggestContours(contours)
        #Get moments of contour; mainly for centroid
        M = cv2.moments(cnt)
        #Get convex hull (bounding polygon on contour)
        hull = cv2.convexHull(cnt)
        #Calculate Contour area
        cntArea = cv2.contourArea(cnt)
        #calculate area of convex hull
        hullArea = cv2.contourArea(hull)
        #Filters contours based off of size
        if (checkContours(cntArea, hullArea)):
            #Gets the centeroids of contour
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            #Gets rotated bounding rectangle of contour
            rect = cv2.minAreaRect(cnt)
            #Creates box around that rectangle
            box = cv2.boxPoints(rect)
            #Not exactly sure
            box = np.int0(box)
            #Gets center of rotated rectangle
            center = rect[0]
            #Gets rotation of rectangle; same as rotation of contour
            rotation = rect[2]
            #Gets width and height of rotated rectangle
            width = rect[1][0]
            height = rect[1][1]
            #Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
            rotation = translateRotation(rotation, width, height)
            #Gets smaller side
            if width > height:
                smaller_side = height
            else:
                smaller_side = width
            #Calculates yaw of contour (horizontal position in degrees)
            yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
            #Calculates yaw of contour (horizontal position in degrees)
            pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)
            #
            #Adds padding for text
            padding  = -8 - math.ceil(.5*smaller_side)
            #Draws rotated rectangle
            cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

            #Draws a vertical white line passing through center of contour
            cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
            #Draws a white circle at center of contour
            cv2.circle(image, (cx, cy), 6, (255, 255, 255))
            #Puts the rotation on screen
            cv2.putText(image, "Rotation: " + str(rotation), (cx + 40, cy + padding), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
            #Puts the yaw on screen
            cv2.putText(image, "Yaw: " + str(yaw), (cx+ 40, cy + padding -16), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
            #Puts the Pitch on screen
            cv2.putText(image, "Pitch: " + str(pitch), (cx+ 80, cy + padding -42), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))


            #Draws the convex hull
            #cv2.drawContours(image, [hull], 0, (23, 184, 80), 3)
            #Draws the contours
            cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

            #Gets the (x, y) and radius of the enclosing circle of contour
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            #Rounds center of enclosing circle
            center = (int(x), int(y))
            #Rounds radius of enclosning circle
            radius = int(radius)
            #Makes bounding rectangle of contour
            rx, ry, rw, rh = cv2.boundingRect(cnt)

            #Draws countour of bounding rectangle and enclosing circle in green
            cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)
            cv2.circle(image, center, radius, (23, 184, 80), 1)

            sendImportantContourInfo(cx, cy, yaw, pitch, rotation, cntArea);
#Draws and calculates contours and their properties
def processContours(contours, image, centerX, centerY):

    #Loop through all contours
    for cnt in contours:
        #Get moments of contour; mainly for centroid
        M = cv2.moments(cnt)
        #Get convex hull (bounding polygon on contour)
        hull = cv2.convexHull(cnt)
        #Calculate Contour area
        cntArea = cv2.contourArea(cnt)
        #calculate area of convex hull
        hullArea = cv2.contourArea(hull)
        #Filters contours based off of size
        if (checkContours(cntArea, hullArea)):
            #Gets the centeroids of contour
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            #Gets rotated bounding rectangle of contour
            rect = cv2.minAreaRect(cnt)
            #Creates box around that rectangle
            box = cv2.boxPoints(rect)
            #Not exactly sure
            box = np.int0(box)
            #Gets center of rotated rectangle
            center = rect[0]
            #Gets rotation of rectangle; same as rotation of contour
            rotation = rect[2]
            #Gets width and height of rotated rectangle
            width = rect[1][0]
            height = rect[1][1]
            #Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
            rotation = translateRotation(rotation, width, height)
            #Gets smaller side
            if width > height:
                smaller_side = height
            else:
                smaller_side = width
            #Calculates yaw of contour (horizontal position in degrees)
            yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
            #Calculates yaw of contour (horizontal position in degrees)
            pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)
            #
            #Adds padding for text
            padding  = -8 - math.ceil(.5*smaller_side)
            #Draws rotated rectangle
            cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

            #Draws a vertical white line passing through center of contour
            cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
            #Draws a white circle at center of contour
            cv2.circle(image, (cx, cy), 6, (255, 255, 255))
            #Puts the rotation on screen
            cv2.putText(image, "Rotation: " + str(rotation), (cx + 40, cy + padding), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
            #Puts the yaw on screen
            cv2.putText(image, "Yaw: " + str(yaw), (cx+ 40, cy + padding -16), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
            #Puts the Pitch on screen
            cv2.putText(image, "Pitch: " + str(pitch), (cx+ 80, cy + padding -42), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))


            #Draws the convex hull
            #cv2.drawContours(image, [hull], 0, (23, 184, 80), 3)
            #Draws the contours
            cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

            #Gets the (x, y) and radius of the enclosing circle of contour
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            #Rounds center of enclosing circle
            center = (int(x), int(y))
            #Rounds radius of enclosning circle
            radius = int(radius)
            #Makes bounding rectangle of contour
            rx, ry, rw, rh = cv2.boundingRect(cnt)

            #Draws countour of bounding rectangle and enclosing circle in green
            cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)
            cv2.circle(image, center, radius, (23, 184, 80), 1)

            sendImportantContourInfo(cx, cy, yaw, pitch, rotation, cntArea);

def findBiggestContours(contours):
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        return c
    else:
        return None;

#Draws contours on blank, black image
def findContoursNewImage(cap, mask, newImage):
    # Finds contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Take each frame
    _, frame = cap.read()
    # Flips the frame so my right is the image's right (probably need to change this
    #frame = cv2.flip(frame, 1)
    # Gets the shape of video
    screenHeight, screenWidth, channels = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    #makes image equal to the assigned image
    image = newImage
    #Draws contours and other stuff
    processLargestContour(contours, image, centerX, centerY)
    #Draws on a blank image
    cv2.imshow("Contours New", image)

#Checks if contours are worthy based off of contour area and (not currently) hull area
def checkContours(cntSize, hullSize):
    if(cntSize > 10000):
        return True
    else:
        return False;


def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return rotation
def calculateDistance(heightOfCamera, heightOfTarget, pitch):
    heightOfCameraFromTarget = heightOfTarget - heightOfCamera

    #Uses trig and pitch to find distance to target
    '''
    d = distance
    h = height between camera and target
    a = angle/pitch
    
    tan a = h/d (opposite over adjacent)
    
    d = h / tan a
    
                         .                 
                        /|        
                       / |       
                      /  |h        
                     /a  |       
              camera -----
                       d         
    '''
    distance = math.fabs(heightOfCameraFromTarget / math.tan(math.radians(pitch)))

    return distance
#Uses trig and focal length of camera to find yaw.
#Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculateYaw(pixelX, centerX, hFocalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
    return yaw
#Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculatePitch(pixelY, centerY, vFocalLength):
    pitch = math.degrees(math.atan((pixelY - centerY) / vFocalLength))
    #Just stopped working have to do this:
    pitch *= -1
    return pitch
def sendImportantContourInfo(contourX, contourY, contourYaw, contourPitch, contourRotation, contourArea):
    print("Contour X: " + str(contourX))
    print("Contour Y: " + str(contourY))
    print("Contour Yaw: " + str(contourYaw))
    print("Contour Pitch: " + str(contourPitch))
    print("Contour Rotation: " + str(contourRotation))
    print("Contour Area: " + str(contourArea))





while(True):

    _, frame = cap.read()
    #frame = cv2.flip(frame, 1)
    screenHeight, screenWidth, channels = frame.shape
    print("Screen width" + str(screenWidth))
    #Calculates focal Length-
    focalLength = screenWidth / (2 * math.tan(HFOV/2))
    print("Focal Length " + str(focalLength))
    print("Vertical Focal Length " + str(screenHeight / (2 * math.tan(VFOV/2))))

    threshold = threshold_video(frame)
    findContours(cap, threshold)
    blank_image = np.zeros((screenHeight, screenWidth, 3), np.uint8)
    findContoursNewImage(cap, threshold, blank_image)
    cv2.imshow("Frame", frame)
    #press escape to exit program
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

