#!/usr/bin/env python3
import numpy as np
import os
import cv2
filename = 'video'
frames_per_seconds = 24.0
res = '720p' #1080p





cap = cv2.VideoCapture(0)
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_720p()


VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'MJPG')

,
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'MJPG')

,
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']
video_type_cv2 = get_video_type(filename)
vidwrite = cv2.VideoWriter('testvideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 25,
           (640,480),True)
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
#change_res(4000, 2000# )

while True:
    ret, frame  = cap.read()
    cv2.imshow("Frame", frame)
    vidwrite.write(frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
vidwrite.release()
cap.release()
cv2.destroyAllWindows()
