import argparse
import sys
import os
import time

from utils import *
import cv2
#####################################################################

model_cfg = './cfg/yolov3-face.cfg'
model_weights = './model-weights/yolov3-wider_16000.weights'
stream_src='video2.mp4'
output_dir='outputs'

if not os.path.exists(output_dir):
    print('==> Creating the {} directory...'.format(output_dir))
    os.makedirs(output_dir)

net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
def return_faces(frame):  
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Remove the bounding boxes with low confidence
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    return faces

def update_driver_now(num_faces,frame_process):
    if sum(num_faces[-frame_process:])==0:
        return False
    elif  sum(num_faces[-frame_process:])< frame_process + 1:
        return True
    return True



def _main():
 
    # Get data from the camera
    cap = cv2.VideoCapture(stream_src)
    num_faces=[]    
    i=0
    passenger_now = ref = True 
    frame_after_process = 10
    frame_process =7
     
    while ref:

        ref, frame = cap.read()

        if i % frame_after_process== 0:
            faces= return_faces(frame)
            num_faces.append(len(faces))

            if len(num_faces) > frame_process:
                passenger_prev=passenger_now
                passenger_now = True if sum(num_faces[-frame_process:])>frame_process else False
                print('passenger are in')
                if passenger_prev and not passenger_now:
                    print('passenger went out')
                    driver_now = True
                    while driver_now:
                        print('waiting for driver to go out' )
                        ref, frame = cap.read()
                        if not ref:
                            print('the streaming ends')
                            break

                        if i% frame_process == 0 :                 
                            print ("please driver go out of the car")
                            num_faces.append(len(return_faces(frame)))
                            driver_now = update_driver_now(num_faces,frame_process)
                        i+=1
                
                if  passenger_prev and not passenger_now and not driver_now:
                    print('we are cleaning now!!!')

            
            for (i, (txt, val)) in enumerate([('number of person is:',f'{len(faces)}')]):
                text = '{}: {}'.format(txt, val)
                cv2.putText(frame, text, (10, (i * 20) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

            cv2.imwrite(os.path.join(output_dir, str(time.time())+'.jpg'), frame.astype(np.uint8))

        i+=1
        h,w,c = frame.shape
        cv2.line(frame,(int(w*0.70),0),(int(w*.70),h),(0,255,0),thickness=2)
        cv2.imshow('frame',frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    _main()
