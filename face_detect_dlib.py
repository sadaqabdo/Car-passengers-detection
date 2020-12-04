import numpy as np
import cv2
import dlib 
from imutils import face_utils



cap = cv2.VideoCapture('video2.mp4')
detector = dlib.get_frontal_face_detector()

i=0
ret=True
while(ret):
    ret, frame = cap.read()
    if i%3==0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rectangles = detector(gray, 0)
        # print(rectangles)
        for rect in rectangles:
            rect = face_utils.rect_to_bb(rect)
            print(i,rectangles)
            cv2.rectangle(frame,pt1=(rect[0],rect[1]),pt2=(rect[2]+rect[0],rect[3]+rect[1]),color=(3,3,3),thickness=2,)
    cv2.imshow('frame',frame)
    i+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()