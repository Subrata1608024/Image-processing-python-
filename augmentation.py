from keras.preprocessing.image import ImageDataGenerator
from skimage import io



datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


dataset = []

import numpy as np
import os
from PIL import Image

image_directory = 'C:/Users/User/Desktop/report picture/thresholding/malignant/'
SIZE = 224
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))
        
x = np.array(dataset)

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='C:/Users/User/Desktop/report picture/threshold aug/malignant/',
                          save_prefix='augml'):
    i += 1
    if i > 10:
        break

                              
                              