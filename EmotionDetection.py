'''
Things done in this python file:
- Face and Emotion detection is done in this python file
Things to be done in this python file:
- Agent calls
'''

import ctypes
import cv2
import time
import queue
import argparse
import threading
import concurrent.futures
import numpy as np
import TrainingModels as tm
from threading import Thread
import PersonalAssistant as pa
from keras.models import load_model
import multiprocessing
from multiprocessing import Process, Array, Value
from keras.preprocessing.image import img_to_array

WAKE_WORD = 'hey bro'
height, width = 48, 48
batch_size = 64
validation_data_dir = './fer2013/validation'
label = ''
# wake = multiprocessing.Value('i', 0)
wake = ''
classifier = load_model('./model2.h5')
print('model loaded')

validation_generator = tm.get_datagen(validation_data_dir)
class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
print(classes)

face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


def face_detector_color(img):
    color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(color, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((197, 197), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_color = color[y:y + h, x:x + w]

    try:
        roi_color = cv2.resize(roi_color, (197, 197), interpolation=cv2.INTER_AREA)
    except:
        return (x, w, y, h), np.zeros((197, 197), np.uint8), img
    return (x, w, y, h), roi_color, img


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


def startSoundThread(sentence):
    mus = Thread(target=pa.speak, args=[sentence])
    mus.start()


def emotion_detector(rect, face, image):
    label = ''
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        # make a prediction on the video image, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (rect[0] + int((rect[1] / 2)), rect[2] + 25)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(image, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    return label


def make_prediction(queue=queue.Queue()):
    global label
    global wake
    cap = cv2.VideoCapture(0)
    print('name is: ', threading.current_thread().name)
    print(threading.get_ident())
    while True:
        ret, frame = cap.read()
        rect, face, image = face_detector(frame)
        if wake == 'hey bro':
            label = emotion_detector(rect, face, image)
            queue.put(label)
        cv2.imshow('All', image)
        if cv2.waitKey(1) == 13:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # print('name is: ', threading.current_thread().name)
    # print(threading.get_ident())
    queue = queue.Queue()
    m = Thread(target=make_prediction, args=(queue,))
    m.start()
    time.sleep(3)
    while True:
        print('Listening')
        text = pa.get_audio()
        if text.count(WAKE_WORD) > 0:
            wake = 'hey bro'
            print(f'i am {wake}')
            while wake == 'hey bro':
                time.sleep(4)
                result = queue.get()
                print('dasda', result)
                speak_thread = Thread(target=pa.decision, args=(result.lower(),))
                print('started thread')
                speak_thread.start()
                speak_thread.join()
                end_confirm = Thread(target=pa.speak, args=['is there anything else i can do for you?'])
                end_confirm.start()
                end_confirm.join()
                text = pa.get_audio()
                if text == 'no' or text == 'stop':
                    wake = 'stop bro'
                    break
        elif text == 'stop':
            wake = 'good bye'
            speak_thread = Thread(target=pa.speak, args=[wake])
            speak_thread.start()
            speak_thread.join()
            # break
    # for t in threading.enumerate():
    #         print(t.name)
