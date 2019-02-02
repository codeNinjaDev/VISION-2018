#!/usr/bin/env python3
#----------------------------------------------------------------------------
# Copyright (c) 2018 FIRST. All Rights Reserved.
# Open Source Software - may be modified and shared by FRC teams. The code
# must be accompanied by the FIRST BSD license file in the root directory of
# the project.
#----------------------------------------------------------------------------

import json
import time
import sys

from cscore import CameraServer, VideoSource
from networktables import NetworkTablesInstance
import cv2
import numpy as np
from networktables import NetworkTables
import math

###################### PROCESSING FUNCTIONS OPENCV (Not all are used) ################################

#Angles in radians
image_width = 320
image_height = 240
#Lifecam 3000
diagonalView = math.radians(68.5)
horizontalAspect = 16
verticalAspect = 9

diagonalAspect = math.sqrt(math.pow(horizontalAspect, 2) + math.pow(verticalAspect, 2))
horizontalView = math.atan(math.tan(diagonalView/2) * (horizontalAspect / diagonalAspect)) * 2
verticalView = math.atan(math.tan(diagonalView/2) * (verticalAspect / diagonalAspect)) * 2

H_FOCAL_LENGTH = image_width / (2*math.tan((horizontalView/2)))
V_FOCAL_LENGTH = image_height / (2*math.tan((verticalView/2)))



# Masks the video based on a range of hsv colors
# Takes in a frame, returns a masked frame
def threshold_video(frame):
    # Gets the shape of video
    screenHeight, screenWidth, channels = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    blur = cv2.medianBlur(frame, 5)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # define range of red in HSV
    lower_color = np.array([60,105,34])
    upper_color = np.array([93, 255, 255])
    # hold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # kernel = np.ones((5,5),np.uint8)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # Shows the theshold image in a new window
    # cv2.imshow('threshold', mask.copy())
    # Returns the masked imageBlurs video to smooth out image

    return mask


# Draws contours on blank, black image
def findContoursNewImage(frame, mask, newImage):
    # Finds contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Flips the frame so my right is the image's right (probably need to change this
    # frame = cv2.flip(frame, 1)
    # Gets the shape of video
    screenHeight, screenWidth, channels = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # makes image equal to the assigned image
    image = newImage
    # Draws contours and other stuff
    processLargestContours(contours, image, centerX, centerY)
    # Draws on a blank image
    return image


# Finds the contours from the masked image and displays them on original stream
def findContours(frame, mask):
    # Finds contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Take each frame
    # Flips the frame so my right is the image's right (probably need to change this
    # frame = cv2.flip(frame, 1)
    # Gets the shape of video
    screenHeight, screenWidth, channels = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage) #TODO finding largest
    if len(contours) != 0:
        image = processLargestContours(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image

width_center_of_tape = 5;
yawFromContour = calculateYaw(x, centerX, H_FOCAL_LENGTH);
tan yaw = width_center_of_tape / calculateDistance()
distance = width_center_of_tape / tan_yaw


# Draws and calculates properties of largest contours ###### USED #####
def processLargestContours(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape;
    targets = []

    if len(contours) >= 2:
        #sortedLargeContours = findBiggestContours(contours)
        # todo JUST ADDDED HAVEN'T TESTED!!!!!!
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        biggestCnts = []
        for cnt in cntsSorted:
            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Get convex hull (bounding polygon on contour)
            hull = cv2.convexHull(cnt)
            # Calculate Contour area
            cntArea = cv2.contourArea(cnt)
            # calculate area of convex hull
            hullArea = cv2.contourArea(hull)
            # Filters contours based off of size
            if (checkContours(cntArea, hullArea)):

                # Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0

                # Gets rotated bounding rectangle of contour
                rect = cv2.minAreaRect(cnt)
                # Creates box around that rectangle
                box = cv2.boxPoints(rect)
                # Not exactly sure
                box = np.int0(box)
                # Gets center of rotated rectangle
                center = rect[0]
                # Gets rotation of rectangle; same as rotation of contour
                rotation = rect[2]
                # Gets width and height of rotated rectangle
                width = rect[1][0]
                height = rect[1][1]
                # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
                rotation = translateRotation(rotation, width, height)

                # Gets smaller side
                if width > height:
                    smaller_side = height
                else:
                    smaller_side = width
                # Calculates yaw of contour (horizontal position in degrees)
                yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                # Calculates yaw of contour (horizontal position in degrees)
                pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)

                # Adds padding for text
                padding = -8 - math.ceil(.5 * smaller_side)
                # Draws rotated rectangle
                cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

                # Draws a vertical white line passing through center of contour
                cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                # Draws a white circle at center of contour
                cv2.circle(image, (cx, cy), 6, (255, 255, 255))
                # Puts the rotation on screen
                #cv2.putText(image, "Rotation: " + str(rotation), (cx + 40, cy + padding), cv2.FONT_HERSHEY_COMPLEX, .6,
                            #(255, 255, 255))
                # Puts the yaw on screen
                #cv2.putText(image, "Yaw: " + str(yaw), (cx + 40, cy + padding - 16), cv2.FONT_HERSHEY_COMPLEX, .6,
                            #(255, 255, 255))
                # Puts the Pitch on screen
                #cv2.putText(image, "Pitch: " + str(pitch), (cx + 80, cy + padding - 42), cv2.FONT_HERSHEY_COMPLEX, .6,
                            #(255, 255, 255))

                # Draws the convex hull
                # cv2.drawContours(image, [hull], 0, (23, 184, 80), 3)
                # Draws the contours
                cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                # Gets the (x, y) and radius of the enclosing circle of contour
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                # Rounds center of enclosing circle
                center = (int(x), int(y))
                # Rounds radius of enclosning circle
                radius = int(radius)
                # Makes bounding rectangle of contour
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                boundingRect = cv2.boundingRect(cnt)
                # Draws countour of bounding rectangle and enclosing circle in green
                cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)

                cv2.circle(image, center, radius, (23, 184, 80), 1)
                if (len(biggestCnts) < 6):
                    biggestCnts.append([cx, cy, rotation, cnt])
                sendImportantContourInfo(cx, cy, yaw, pitch, rotation, cntArea);

        biggestCnts = sorted(biggestCnts, key=lambda x: x[0])
        # 2019 target tracking
        for i in range(len(biggestCnts) - 1):
            tilt1 = biggestCnts[i][2]
            tilt2 = biggestCnts[i + 1][2]

            cx1 = biggestCnts[i][0]
            cx2 = biggestCnts[i + 1][0]
            cy1 = biggestCnts[i][1]
            cy2 = biggestCnts[i + 1][1]

            # If contour angles are opposite
            if (np.sign(tilt1) != np.sign(tilt2)):
                centerOfTarget = math.floor((cx1 + cx2) / 2)

                # If left contour rotation is tilted to the left then skip iteration
                if (tilt1 < 0):
                    if (cx1 < cx2):
                        continue
                # If left contour rotation is tilted to the left then skip iteration
                if (tilt2 < 0):
                    if (cx2 < cx1):
                        continue
                # pixelDistanceOfTargetToCenter = round(math.fabs(centerXOfImage - centerOfTarget))
                yawToTarget = calculateYaw(centerOfTarget, centerX, H_FOCAL_LENGTH)
                if [centerOfTarget, yawToTarget] not in targets:
                    targets.append([centerOfTarget, yawToTarget])

    if (len(targets) > 0):
        print(targets)
        targets.sort(key=lambda x: math.fabs(x[0]))
        finalTarget = min(targets, key=lambda x: math.fabs(x[1]))
        print(finalTarget)
        # Puts the yaw on screen
        cv2.putText(image, "Yaw: " + str(finalTarget[1]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
                    (255, 255, 255))
        print("Yaw: " + str(finalTarget[1]))
        cv2.line(image, (finalTarget[0], screenHeight), (finalTarget[0], 0), (255, 0, 0), 2)
    return image
# Draws and calculates contours and their properties
def processContours(contours, image, centerX, centerY):

    screenHeight, screenWidth, channels = image.shape;
    # Loop through all contours
    for cnt in contours:
        # Get moments of contour; mainly for centroid
        M = cv2.moments(cnt)
        # Get convex hull (bounding polygon on contour)
        hull = cv2.convexHull(cnt)
        # Calculate Contour area
        cntArea = cv2.contourArea(cnt)
        # calculate area of convex hull
        hullArea = cv2.contourArea(hull)
        # Filters contours based off of size
        if (checkContours(cntArea, hullArea)):
            # Gets the centeroids of contour
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Gets rotated bounding rectangle of contour
            rect = cv2.minAreaRect(cnt)
            # Creates box around that rectangle
            box = cv2.boxPoints(rect)
            # Not exactly sure
            box = np.int0(box)
            # Gets center of rotated rectangle
            center = rect[0]
            # Gets rotation of rectangle; same as rotation of contour
            rotation = rect[2]
            # Gets width and height of rotated rectangle
            width = rect[1][0]
            height = rect[1][1]
            # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
            rotation = translateRotation(rotation, width, height)
            # Gets smaller side
            if width > height:
                smaller_side = height
            else:
                smaller_side = width
            # Calculates yaw of contour (horizontal position in degrees)
            yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
            # Calculates yaw of contour (horizontal position in degrees)
            pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)

            padding = -8 - math.ceil(.5 * smaller_side)
            # Draws rotated rectangle
            cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

            # Draws a vertical white line passing through center of contour
            cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
            # Draws a white circle at center of contour
            cv2.circle(image, (cx, cy), 6, (255, 255, 255))
            # Puts the rotation on screen
            cv2.putText(image, "Rotation: " + str(rotation), (cx + 40, cy + padding), cv2.FONT_HERSHEY_COMPLEX, .6,
                        (255, 255, 255))
            # Puts the yaw on screen
            cv2.putText(image, "Yaw: " + str(yaw), (cx + 40, cy + padding - 16), cv2.FONT_HERSHEY_COMPLEX, .6,
                        (255, 255, 255))
            # Puts the Pitch on screen
            cv2.putText(image, "Pitch: " + str(pitch), (cx + 80, cy + padding - 42), cv2.FONT_HERSHEY_COMPLEX, .6,
                        (255, 255, 255))

            # Draws the convex hull
            # cv2.drawContours(image, [hull], 0, (23, 184, 80), 3)
            # Draws the contours
            cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

            # Gets the (x, y) and radius of the enclosing circle of contour
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            # Rounds center of enclosing circle
            center = (int(x), int(y))
            # Rounds radius of enclosning circle
            radius = int(radius)
            # Makes bounding rectangle of contour
            rx, ry, rw, rh = cv2.boundingRect(cnt)

            # Draws countour of bounding rectangle and enclosing circle in green
            cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)
            cv2.circle(image, center, radius, (23, 184, 80), 1)
            sendImportantContourInfo(cx, cy, yaw, pitch, rotation, cntArea);


# Checks if contours are worthy based off of contour area and (not currently) hull area
def checkContours(cntSize, hullSize):
    if (cntSize >= 10):
        return True
    else:
        return False;


def findBiggestContours(contours):
    biggestContours = []
    while len(contours) > 0:
        c = max(contours, key = cv2.contourArea)
        biggestContours.append(c)
        contours.remove(c)
    return biggestContours



def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)


def calculateDistance(heightOfCamera, heightOfTarget, pitch):
    heightOfCameraFromTarget = heightOfTarget - heightOfCamera

    # Uses trig and pitch to find distance to target
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


# Uses trig and focal length of camera to find yaw.
# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculateYaw(pixelX, centerX, hFocalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
    return round(yaw)


# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculatePitch(pixelY, centerY, vFocalLength):
    pitch = math.degrees(math.atan((pixelY - centerY) / vFocalLength))
    # Just stopped working have to do this:
    pitch *= -1
    return round(pitch)


def sendImportantContourInfo(contourX, contourY, contourYaw, contourPitch, contourRotation, contourArea):
    #print("Contour X: " + str(contourX))
    #print("Contour Y: " + str(contourY))
    #print("Contour Yaw: " + str(contourYaw))
    #print("Contour Pitch: " + str(contourPitch))
    #print("Contour Rotation: " + str(contourRotation))
    #print("Contour Area: " + str(contourArea))
    pass

#################### FRC VISION PI Image Specific #############
configFile = "/boot/frc.json"

class CameraConfig: pass

team = None
server = False
cameraConfigs = []

"""Report parse error."""
def parseError(str):
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

"""Read single camera configuration."""
def readCameraConfig(config):
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    cam.config = config

    cameraConfigs.append(cam)
    return True

"""Read configuration file."""
def readConfig():
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    return True

"""Start running the camera."""
def startCamera(config):
    print("Starting camera '{}' on {}".format(config.name, config.path))
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)

    camera.setConfigJson(json.dumps(config.config))

    return cs, camera

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]
    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClientTeam(team)

    # start cameras
    cameras = []
    streams = []
    for cameraConfig in cameraConfigs:
        cs, cameraCapture = startCamera(cameraConfig)
        streams.append(cs)
        cameras.append(cameraCapture)
    #Get the first camera
    cameraServer = streams[0]
    # Get a CvSink. This will capture images from the camera
    cvSink = cameraServer.getVideo()

    # (optional) Setup a CvSource. This will send images back to the Dashboard
    outputStream = cameraServer.putVideo("stream", image_width, image_height)
    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)

    # loop forever
    while True:
        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error notify the output.
        timestamp, img = cvSink.grabFrame(img)
        frame = img
        if timestamp == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError());
            # skip the rest of the current iteration
            continue

        #
        # Insert your image processing logic here!
        #

        threshold = threshold_video(img)
        processed = findContours(frame, threshold)
        # (optional) send some image back to the dashboard
        outputStream.putFrame(processed)









'''
while(True):

    _, frame = cap.read()
    #frame = cv2.flip(frame, 1)
    screenHeight, screenWidth, channels = frame.shape
    print("Screen width" + str(screenWidth))
    #Calculates focal Length-
    #tan of a
    tanA = (math.tan(math.radians(68.3)))
    #cos of atan(h/2)
    cosA = math.cos(math.atan2(16, 9))
    aH = math.degrees(2*math.atan(cosA * tanA))
    print(aH)
    
    blank_image = np.zeros((screenHeight, screenWidth, 3), np.uint8)
    findContoursNewImage(cap, threshold, blank_image)
    cv2.imshow("Frame", frame)
    #press escape to exit program
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
'''
