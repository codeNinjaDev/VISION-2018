#!/usr/bin/env python3

import cv2
cap = cv2.VideoCapture(1)



pictureCount = 0
while cap.isOpened():
    cap.set(3, 1024)  # width=1280
    cap.set(4, 768)  # height=720
    _,frame = cap.read()

    if _ and frame is not None:
        cv2.imshow('smallimg', frame)  # display the captured image
        if cv2.waitKey(1) & 0xFF == ord('q'):  # exit on pressing 'q'
            pictureCount = pictureCount + 1
            pictureCountString = "smallerimg" + str(pictureCount) + ".png"
            print(pictureCountString)
            print(frame.shape[:2][::-1])
            cv2.imwrite(pictureCountString, frame)
            cv2.destroyAllWindows()


cap.release()
