import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2
from PIL import Image

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.1, 0.9]
)

os.mkdir('newImages')
for file in os.listdir('images/'):
    os.mkdir('newImages/'+file)
    img = load_img('images/'+file)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape) 
    i=0
    for batch in datagen.flow(x, batch_size = 1, save_to_dir='newImages/'+file, save_prefix=file, save_format='jpg'):
        i += 1 
        if i > 3: break
    
    
for file in os.listdir('images/'):
    img = load_img('images/'+file)
    width, height = img.size
    for i in range(0, 4):
        if i==0:
            box_tupple = (0, 0,  width/2, height/2)
        if i==1:
            box_tupple = (width/2, 0, width, height/2)
        if i==2:
            box_tupple = (0, height/2, width/2, height)
        if i==3:
            box_tupple = (width/2, height/2, width, height)

        img_cropped = img.crop(box_tupple)
        imgx = np.asarray(img_cropped)
        imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join('newImages/'+file+'/', file+'_mosaic'+str(i+1)+'.jpg'), imgx)
    