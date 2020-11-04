from __future__ import print_function
import os
import cv2
import keras
import numpy as np
from time import sleep
from keras import layers
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.regularizers import l2
from keras.models import Sequential
from keras.models import load_model
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization

num_features = 64
num_classes = 6
height, width = 48, 48
epochs = 100
batch_size = 64

train_data_dir = './fer2013/train'
validation_data_dir = './fer2013/validation'

# Let's use some data augmentaiton
train_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0,
    zoom_range=0.0,
    horizontal_flip=True,
    vertical_flip=False)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# def get_model1():
#   model = Sequential()
#
#   model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(48,48,1)))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.25))
#
#   model.add(Convolution2D(128, (3, 3), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.25))
#
#   model.add(Convolution2D(256, (3, 3), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.25))
#
#   model.add(Convolution2D(512, (3, 3), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.25))
#
#   model.add(Flatten())
#
#   model.add(Dense(512))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Dropout(0.25))
#
#   model.add(Dense(256))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Dropout(0.25))
#
#   model.add(Dense(6))
#   model.add(Activation('softmax'))
#
#   return model
#
# # actual model
# def get_model2():
#   model = Sequential()
#
#   model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(48,48,1)))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.25))
#
#   model.add(Convolution2D(128, (5, 5), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.25))
#
#   model.add(Convolution2D(512, (3, 3), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.25))
#
#   model.add(Convolution2D(512, (3, 3), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.25))
#
#   model.add(Flatten())
#
#   model.add(Dense(256))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Dropout(0.25))
#
#   model.add(Dense(512))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Dropout(0.25))
#
#   model.add(Dense(6))
#   model.add(Activation('softmax'))
#
#   return model
#
# # deeper model
# def get_model3():
#   model = Sequential()
#
#   model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(48,48,1)))
#   model.add(Convolution2D(64, (3, 3), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.4))
#
#   model.add(Convolution2D(128, (3, 3), padding='same'))
#   model.add(Convolution2D(128, (3, 3), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.4))
#
#   model.add(Convolution2D(256, (3, 3), padding='same'))
#   model.add(Convolution2D(265, (3, 3), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.4))
#
#   model.add(Convolution2D(512, (3, 3), padding='same'))
#   model.add(Convolution2D(512, (3, 3), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.4))
#
#   model.add(Flatten())
#
#   model.add(Dense(2048))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Dropout(0.4))
#
#   model.add(Dense(512))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Dropout(0.4))
#
#   model.add(Dense(6))
#   model.add(Activation('softmax'))
#
#   return model
#
# # shallow model
# def get_model4():
#   model = Sequential()
#
#   model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(48,48,1)))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.25))
#
#   model.add(Convolution2D(128, (5, 5), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
#   model.add(Dropout(0.25))
#
#   model.add(Flatten())
#
#   model.add(Dense(256))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Dropout(0.25))
#
#   model.add(Dense(6))
#   model.add(Activation('softmax'))
#
#   return model
#
# model1 = get_model1()
# model2 = get_model2()
# model3 = get_model3()
# model4 = get_model4()
#
# model1.summary()
# model2.summary()
# model3.summary()
# model4.summary()

# checkpoint = ModelCheckpoint("./emotion.h5",
#                              monitor="val_loss",
#                              mode="min",
#                              save_best_only=True,
#                              verbose=1)
#
# earlystop = EarlyStopping(monitor='val_loss',
#                           min_delta=0,
#                           patience=20,
#                           verbose=0,
#                           restore_best_weights=True)
#
# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss', factor=0.1,
#     patience=10, verbose=0, mode='auto',
#     min_delta=0.0001, cooldown=0, min_lr=0)
#
# # we put our call backs into a callback list
# callbacks = [earlystop, checkpoint, reduce_lr]
#
# model1.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy'])
#
# model2.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy'])
#
# model3.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy'])
#
# model4.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy'])
#
# nb_train_samples = 28273
# nb_validation_samples = 3534
# epochs = 50
#
# history1 = model1.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     verbose=1,
#     callbacks=callbacks,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)
#
# history2 = model2.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     verbose=1,
#     callbacks=callbacks,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)
#
# history3 = model3.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     verbose=1,
#     callbacks=callbacks,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)
#
# history4 = model4.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     verbose=1,
#     callbacks=callbacks,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)

classifier = load_model('./e.h5')
print('model loaded')

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode = 'grayscale',
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
print(class_labels)

face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    except:
        return (x, w, y, h), np.zeros((48, 48), np.uint8), img
    return (x, w, y, h), roi_gray, img


def make_prediction():
    cap = cv2.VideoCapture(0)
    confirmation = input('type yes if you want to detect the expressions: ')
    while True:
        ret, frame = cap.read()
        rect, face, image = face_detector(frame)
        if confirmation == 'y':
            emotion_detector(rect,face,image)
        # if np.sum([face]) != 0.0:
        #     roi = face.astype("float") / 255.0
        #     roi = img_to_array(roi)
        #     roi = np.expand_dims(roi, axis=0)
        #
        #     # make a prediction on the ROI, then lookup the class
        #     preds = classifier.predict(roi)[0]
        #     label = class_labels[preds.argmax()]
        #     print(label)
        #     label_position = (rect[0] + int((rect[1] / 2)), rect[2] + 25)
        #     cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        # else:
        #     cv2.putText(image, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            cv2.imshow('All', image)
            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break
        else:
            cv2.imshow('All', image)
            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break
    cap.release()
    cv2.destroyAllWindows()


def emotion_detector(rect, face, image):
    label = ''
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        print(label)
        label_position = (rect[0] + int((rect[1] / 2)), rect[2] + 25)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(image, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    return label


make_prediction()