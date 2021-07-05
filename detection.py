import cv2
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('.')

font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

cap =cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    frame = cv2.resize(frame,(256,256))
    x = frame.tolist()
    x = np.array(x)
    x = x.reshape(-1, 256, 256, 3)
    x = tf.convert_to_tensor(x, np.float32)
    #print(x)
    output = model.predict(x)
    print(output)
    if output==1:
        cv2.putText(frame, 'Hand', org, font,
                   fontScale, color, thickness, cv2.LINE_AA)
    elif output==0:
        cv2.putText(frame, 'Fist', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    else:
        continue
    cv2.imshow('abc',frame)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
