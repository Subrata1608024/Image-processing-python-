import numpy as np
np.random.seed(1000)
import cv2
import os
from PIL import Image
import keras

os.environ['KERAS_BACKEND'] = 'tensorflow'


image_directory = 'E:/image/threshold segmentation'
SIZE = 64
dataset = []
label = []

benign_images = os.listdir(image_directory +'/benign/')
for i, image_name in enumerate(benign_images):
    if(image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory +'/benign/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))
        label.append(0)
        
        
malignant_images = os.listdir(image_directory +'/malignant/')
for i, image_name in enumerate(malignant_images):
    if(image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory +'/malignant/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))
        label.append(1)
        





###############################
INPUT_SHAPE = (64,64,3)
inp = keras.layers.Input(shape=INPUT_SHAPE)

conv1 = keras.layers.Conv2D(32, kernel_size=(3,3), activation = 'relu' , padding='same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = keras.layers.BatchNormalization(axis=-1)(pool1)
drop1 = keras.layers.Dropout(rate=0.2)(norm1)

conv2 = keras.layers.Conv2D(64, kernel_size=(3,3), activation = 'relu' , padding='same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = keras.layers.BatchNormalization(axis=-1)(pool2)
drop2 = keras.layers.Dropout(rate=0.2)(norm2)

conv3 = keras.layers.Conv2D(128, kernel_size=(3,3), activation = 'relu' , padding='same')(drop2)
pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
norm3 = keras.layers.BatchNormalization(axis=-1)(pool3)
drop3 = keras.layers.Dropout(rate=0.2)(norm3)

conv4 = keras.layers.Conv2D(256, kernel_size=(3,3), activation = 'relu' , padding='same')(drop3)
pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
norm4 = keras.layers.BatchNormalization(axis=-1)(pool4)
drop4 = keras.layers.Dropout(rate=0.2)(norm4)


conv5 = keras.layers.Conv2D(256, kernel_size=(3,3), activation = 'relu' , padding='same')(drop4)
pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)
norm5 = keras.layers.BatchNormalization(axis=-1)(pool5)
drop5 = keras.layers.Dropout(rate=0.2)(norm5)



flat = keras.layers.Flatten()(drop5)

hidden1 = keras.layers.Dense(512, activation = 'relu')(flat)
norm6 = keras.layers.BatchNormalization(axis=-1)(hidden1)
drop6 = keras.layers.Dropout(rate=0.5)(norm6)

hidden2 = keras.layers.Dense(512, activation = 'relu')(flat)
norm7 = keras.layers.BatchNormalization(axis=-1)(hidden2)
drop7 = keras.layers.Dropout(rate=0.5)(norm7)



out = keras.layers.Dense(2, activation='sigmoid')(drop7)

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam' , 
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
print(model.summary())




#################################

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.20, random_state = 42)
history = model.fit(np.array(X_train), y_train, batch_size = 16, verbose=1, epochs=50, validation_split=0.1, shuffle=False)

print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))

import matplotlib.pyplot as plt
f,(ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.1)

max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")








