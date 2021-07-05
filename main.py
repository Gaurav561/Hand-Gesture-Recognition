import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataset import create_dataset
import numpy as np

x_train,y_train = create_dataset(200,200)
x_test,y_test = create_dataset(40,40)

x_train = np.array(x_train)


x_train = tf.convert_to_tensor(x_train, np.float32)
y_train = tf.convert_to_tensor(y_train, np.float32)


x_test = tf.convert_to_tensor(x_test, np.float32)
y_test = tf.convert_to_tensor(y_test, np.float32)



#x_train = x_train.reshape(40, 256, 256, 3)

#x_train = tf.expand_dims(x_train, axis=-1)
#x_train = tf.reshape(x_train, (40, 256, 256, 3))
#x_train = np.expand_dims(x_train, axis=-1)
#x_train = x_train[..., np.newaxis]

print(np.shape(x_train))
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))


model  = keras.models.Sequential()


#x_train = np.array(x_train,dtype=np.uint8)
#y_train = np.array(y_train,dtype=np.uint8)
#x_test = np.array(x_test,dtype=np.uint8)
#x_test = np.array(y_test,dtype=np.uint8)


model.add(layers.Conv2D(64,(3,3),input_shape=(256,256,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(16,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


model.summary()


model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
model.fit(x_train,y_train,epochs=30,batch_size=10,validation_data=(x_test,y_test))

model.save('.')
