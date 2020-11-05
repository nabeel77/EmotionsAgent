from __future__ import print_function
import os
import ray
import cv2
import time
import keras
import queue
import numpy as np
from time import sleep
from keras import layers
import concurrent.futures
from threading import Thread
from keras.models import Model
from keras.layers import Input
import PersonalAssistant as pa
from keras.layers import Flatten
from keras.regularizers import l2
from keras.models import Sequential
from keras.models import load_model
from multiprocessing import Process
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
from multiprocessing import Process
import ctypes
import argparse
from multiprocessing import Array, Value, Process

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

name = 'Hello Nabeel, how are you'

# def playSound():
#     pa.speak('Hello {}, How are you'.format(name))

def startSoundThread():
         mus = Thread(target=pa.speak, args=[name])
         mus.start()
         # q = queue.Queue()
         # listener = Thread(target=pa.get_audio, args=(q,)).start()
         # result = q.get()
         # print(result)

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     audioThread = executor.submit(playSound)

def make_prediction():
    cap = cv2.VideoCapture(0)
    #confirmation = input('type yes if you want to detect the expressions: ')
    counter = 1
    while cap.isOpened():
        while True:
            ret, frame = cap.read()
            rect, face, image = face_detector(frame)
            emotion_detector(rect,face,image)
            Show_cam(image)
            if counter == 1:
                startSoundThread()
                #text = pa.get_audio()
                #print(text)
                counter = -1
            # if counter == -1:
            #     q = queue.Queue()
            #     listener = Thread(target=pa.get_audio, args=(q,)).start()
            #     result = q.get()
            #     print(result)
            #     counter = 2
            #Show_cam(image)
            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break
            # else:
            #     Show_cam(image)
            #     if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            #         break
        cap.release()
        cv2.destroyAllWindows()
    else:
        pa.speak('Camera is disconnected. Good bye.')


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

def Show_cam(image):
    cv2.imshow('All', image)



# pa.speak('hi')x`
# if __name__ == '__main__':
#     make_prediction()
    # mus = Thread(target=playSound)
    # mus.start()

class VideoCapture:
    """
    Class that handles video capture from device or video file
    """
    def __init__(self, device=0, delay=0.):
        """
        :param device: device index or video filename
        :param delay: delay between frame captures in seconds(floating point is allowed)
        """
        self._cap = cv2.VideoCapture(device)
        self._delay = delay

    def _proper_frame(self, delay=None):
        """
        :param delay: delay between frames capture(in seconds)
        :param finished: synchronized wrapper for int(see multiprocessing.Value)
        :return: frame
        """
        snapshot = None
        correct_img = False
        fail_counter = -1
        while not correct_img:
            # Capture the frame
            correct_img, snapshot = self._cap.read()
            fail_counter += 1
            # Raise exception if there's no output from the device
            if fail_counter > 10:
                raise Exception("Capture: exceeded number of tries to capture the frame.")
            # Delay before we get a new frame
            time.sleep(delay)
        return snapshot

    def get_size(self):
        """
        :return: size of the captured image
        """
        return (int(self._cap.get(int(cv2.CAP_PROP_FRAME_HEIGHT))),
                int(self._cap.get(int(cv2.CAP_PROP_FRAME_WIDTH))), 3)

    def get_stream_function(self):
        """
        Returns stream_function object function
        """

        def stream_function(image, finished):
            """
            Function keeps capturing frames until finished = 1
            :param image: shared numpy array for multiprocessing(see multiprocessing.Array)
            :param finished: synchronized wrapper for int(see multiprocessing.Value)
            :return: nothing
            """
            # Incorrect input array
            if image.shape != self.get_size():
                raise Exception("Capture: improper size of the input image")
            print("Capture: start streaming")
            # Capture frame until we get finished flag set to True
            while not finished.value:
                image[:, :, :] = self._proper_frame(self._delay)
            # Release the device
            self.release()

        return stream_function

    def release(self):
        self._cap.release()


def main():
    # Add program arguments
    parser = argparse.ArgumentParser(description='Captures the video from the webcamera and \nwrites it into the output file with predefined fps.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-output', dest="output",  default="output.avi", help='name of the output video file')
    parser.add_argument('-log', dest="log",  default="frames.log", help='name of the log file')
    parser.add_argument('-fps', dest="fps",  default=25., help='frames per second value')

    # Read the arguments if any
    result = parser.parse_args()
    fps = float(result.fps)
    output = result.output
    log = result.log

    # Initialize VideoCapture object and auxilary objects
    cap = VideoCapture()
    shape = cap.get_size()
    stream = cap.get_stream_function()

    # Define shared variables(which are synchronised so race condition is excluded)
    shared_array_base = Array(ctypes.c_uint8, shape[0] * shape[1] * shape[2])
    frame = np.ctypeslib.as_array(shared_array_base.get_obj())
    frame = frame.reshape(shape[0], shape[1], shape[2])
    finished = Value('i', 0)

    # Start processes which run in parallel
    video_process = Process(target=stream, args=(frame, finished))
    video_process.start()  # Launch capture process

    # Sleep for some time to allow videocapture start working first
    time.sleep(2)

    # Termination function
    def terminate():
        print("Main: termination")
        finished.value = True
        # Wait for all processes to finish
        time.sleep(1)
        # Terminate working processes
        video_process.terminate()

    # The capturing works until keyboard interrupt is pressed.
    counter = 1
    while True:
        try:
            # Display the resulting frame
            rect, face, image = face_detector(frame)
            emotion_detector(rect, face, image)
            if counter == 1:
                startSoundThread()
                p = Thread(target=pa.get_audio)
                p.start()
                counter = -1
            cv2.imshow('frame', frame)
            cv2.waitKey(1)  # Display it at least one ms before going to the next frame
            time.sleep(0.1)

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            terminate()
            break

if __name__ == '__main__':
    main()
