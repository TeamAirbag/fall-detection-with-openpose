import argparse
import logging
import time
import math
import datetime
import pygame
import pandas as pd
import keyboard
from pprint import pprint
from sklearn.svm import SVC
import cv2
import numpy as np
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import os
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--save_video',type=bool,default=False, 
                        help='To write output video. default name file_name_output.avi')
    args = parser.parse_args()
    
    print("mode 0: Only Pose Estimation \nmode 1: People Counter \nmode 2: Fall Detection")
    mode = int(input("Enter a mode : "))
    
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    ##########################################################################################
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    # if(args.camera == '0'):
    #     file_write_name = 'camera_0'
    # else:
    #     #basename = os.path.basename(args.camera)
    #     # path = os.path.dirname(imgfile)
    #     file_write_name, _ = os.path.splitext(args.camera) 
    ret_val, image = cam.read()
    image = cv2.resize(image, dsize=(300, 300), interpolation=cv2.INTER_AREA)
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    count = 0
    y1 = [0,0]
    frame = 0

    dataset = []
    state = []          


    ### 모델 학습
    data = pd.read_csv("posedata.csv")
    X, Y = data.iloc[:,:36], data['class']
    x = X.to_numpy()
    y = Y.to_numpy()
    model = SVC(kernel='poly')
    model.fit(x, y)


    while True:
        ret_val, image = cam.read()
        image = cv2.resize(image, dsize=(300, 300), interpolation=cv2.INTER_AREA)
        i =1
        count+=1
        if count % 11 == 0:
            continue
        # logger.debug('image process+')
        if not ret_val:
            break
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        # In humans total num of detected person in frame
        # logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        # logger.debug('show+')
        if mode == 1:
            hu = len(humans)
            print("Total no. of People : ", hu)
        elif mode == 2:
            for human in humans:
                # we select one person from num of person
                for i in range(len(humans)):
                    try:
                        '''
                        To detect fall we have used y coordinate of head.
                        Coordinates of head in form of normalize form.
                        We convert normalized points to relative point as per the image size.
                        y1.append(y) will store y coordinate to compare with previous point.
                        We have used try and except because many time pose estimator cann't predict head point.
                        
                        '''
                        # human.parts contains all the detected body parts
                        a = human.body_parts[0]  # human.body_parts[0] is for head point coordinates
                        x = a.x*image.shape[1]   # x coordinate relative to image 
                        y = a.y*image.shape[0]   # y coordinate relative to image
                        y1.append(y)   # store value of y coordinate in list to compare two frames

                        ### x1=[]
                        ### for j in range (0, 18):
                        ###    print('j=', j, '=', human.body_parts[j].x)
                        ###   x1.append(human.body_parts[j].x)
                        ###print(x1)
                    
                        min_x,min_y,max_x,max_y=image.shape[1],image.shape[0],0,0
                        

                        # for BodyPart in human.body_parts.values():
                            
                        #     x=BodyPart.x
                        #     y=BodyPart.y

                        #     # x = x*image.shape[1]   # x coordinate relative to image 
                        #     # y = y*image.shape[0]   # y coordinate relative to image

                        #     min_x=min(min_x,x)
                        #     min_y=min(min_y,y)
                        #     max_x=max(max_x,x)
                        #     max_y=max(max_y,y)
                        # # print(min_x, min_y, max_x, max_y)

                        # rec = (max_x-min_x)/(max_y-min_y)



                        # ### 데이터 저장
                        # col_name = ['nose_x', 'nose_y', 'neck_x', 'neck_y', 'Rshoulder_x', 'Rshoulder_y', 'Relbow_x', 'Relbow_y', 'Rwrist_x', 'Rwrist_y',
                        # 'Lshoulder_x', 'Lshoulder_y', 'Lelbow_x', 'Lelbow_y', 'Lwrist_x', 'Lwrist_y', 'Rhip_x', 'Rhip_y', 'Rknee_x', 'Rknee_y',
                        # 'Rankle_x', 'Rankle_y', 'Lhip_x', 'Lhip_y', 'Lknee_x', 'Lknee_y', 'Lankle_x', 'Lankle_y', 'Reye_x', 'Reye_y',
                        # 'Leye_x', 'Leye_y', 'Rear_x', 'Rear_y', 'Lear_x', 'Lear_y', 'class']

                        # x1 = [0 for i in range(0,36)]
                        # x1.append(0)  # 클래스, (stand : 0, sit:1, lie:2)
                        # for j in range(0,34):
                        #     if human.body_parts[j].x != None:
                        #         x1[2*j] = human.body_parts[j].x
                        #         x1[2*j+1] = human.body_parts[j].y
                        #     dataset.append(x1)
                        #     print(x1)
                        #     if keyboard.is_pressed('q'):
                        #         print('데이터 저장 DEMO')
                        #         df_demo = pd.DataFrame(dataset, columns=col_name)
                        #         df_demo.to_csv('C:\\Users\\dongik\\Desktop\\tf-pose-estimation-master\\data\\stand3.csv', sep=',')

                        ## 실시간 좌표값 기반 모델 예측
                        x1 = [0 for i in range(0,36)]       # 실시간 좌표값
                        for j in range(0,34):
                            if human.body_parts[j].x != None:
                                x1[2*j] = human.body_parts[j].x
                                x1[2*j+1] = human.body_parts[j].y
                            result = model.predict([x1])
                            if result == [0]:               # stand
                                tt = "stand"
                                state.append(0)
                                begin = time.time()
                            elif result == [1]:             # sit
                                tt = "sit"
                                state.append(1)
                                begin = time.time()                                  
                            elif result == [2]:             # lie
                                tt = "lie"
                                if len(state) != 0:
                                    end = time.time()
                                    if state[-1] == 0 or state[-1] == 1:   # 전 값이 서있는 상태 또는 앉아있는 상태
                                        if (end-begin) <= 2:
                                            print("fall이 출력되어야함")
                                            tt = "fall"
                                state = []
                            elif result == [3]:             # normal
                                tt = "normal"
                                state = []

                    except:
                        pass
                    cv2.rectangle(image, (int(min_x), int(max_y)), (int(max_x), int(min_y)), (255,0,0), 1)

                    # if ((y - y1[-2]) > 25):  # it's distance between frame and comparing it with thresold value 
                    #     cv2.putText(image, "Fall Detected", (20,50), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0,0,255), 
                    #         2, 11)
                    #     print("fall detected.",i+1, count) # You can set count for get that your detection is working
    ###################################################################################################################
        elif mode == 0:	
        	pass
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        cv2.putText(image, tt, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,0,0), 3)

        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        # if(frame == 0) and (args.save_video):   # It's use to intialize video writer ones
        #     out = cv2.VideoWriter(file_write_name+'_output.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        #             20,(image.shape[1],image.shape[0]))
        # out.write(image)
        if cv2.waitKey(1) == 27:
            break
        # logger.debug('finished+')

    cv2.destroyAllWindows()