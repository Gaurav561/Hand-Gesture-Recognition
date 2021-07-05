import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

def create_dataset(n,m):
    cap = cv2.VideoCapture(0)
    data1 = []
    i=0
    while i<n:
        _,frame = cap.read()

        frame = cv2.resize(frame,(256,256))

        data1.append([frame,1])
        cv2.imshow('abc',frame)
        i=i+1
        k = cv2.waitKey(10)
        if k == ord('q'):
            break


    cv2.destroyAllWindows()
    print('************************************************************************************************************************')
    cv2.waitKey(2000)
    data2 = []
    i=0
    while i<m:
        _,frame = cap.read()

        frame = cv2.resize(frame,(256,256))

        data2.append([frame,0])
        cv2.imshow('efg',frame)
        i=i+1
        k = cv2.waitKey(10)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #data1=np.array(data1,dtype=np.float)
    #data2=np.array(data2,dtype=np.float)

    data = np.concatenate((data1,data2))
    np.random.shuffle(data)
    x=data[:,0]
    x = x/255.0
    y=data[:,1]
    x = x.tolist()
    y = y.tolist()
    #x=np.asarray(x).astype(np.int)
    #y=np.asarray(y).astype(np.int)
    return x,y
