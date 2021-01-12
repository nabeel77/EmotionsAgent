'''
Things done in this python file:
- Face and Emotion detection is done in this python file
Things to be done in this python file:
- Agent calls
'''

import cv2
import time
import queue
import numpy as np
import TrainingModels as tm
from threading import Thread
import PersonalAssistant as pa
from keras.models import load_model
from keras.preprocessing.image import img_to_array

WAKE_WORD = 'hey assistant'
height, width = 48, 48
batch_size = 64
validation_data_dir = './fer2013/validation'
label = ''
wake = ''
classifier = load_model('./ResNet50.h5')

validation_generator = tm.get_datagen(validation_data_dir)
class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())

face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


def face_detector(img, color=cv2.COLOR_BGR2GRAY, size=48):
    '''
    :param img: img file in which the face needs to be detected
    :param color: default value is cv2.COLOR_BGR2GRAY. cv2.COLOR_BGR2RGB should be passed if emotion detection
                  is being done by ResNet50.
    :param size: default value is 48, 197 should be passed if emotion detection is being done by using Resnet50
    :return: return the image file with the triangle around the detected face.
    '''

    gray = cv2.cvtColor(img, color)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((size, size), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

    try:
        roi_gray = cv2.resize(roi_gray, (size, size), interpolation=cv2.INTER_AREA)
    except:
        return (x, w, y, h), np.zeros((size, size), np.uint8), img
    return (x, w, y, h), roi_gray, img


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


def make_prediction(q=queue.Queue()):
    global label
    global wake
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        rect, face, image = face_detector(frame, color=cv2.COLOR_BGR2RGB, size=197)
        if wake == 'hey assistant':
            label = emotion_detector(rect, face, image)
            q.put(label)
        cv2.imshow('Intelligent Agent', image)
        if cv2.waitKey(1) == 13 :
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    queue = queue.Queue()
    emotion_detection_thread = Thread(target=make_prediction, args=(queue,))
    emotion_detection_thread.start()
    time.sleep(5)
    while True:
        print('Listening')
        text = pa.get_audio()
        if text.count(WAKE_WORD) > 0:
            wake = 'hey assistant'
            while wake == 'hey assistant':
                time.sleep(4)
                emotion = queue.get()
                agent_thread = Thread(target=pa.decision, args=(emotion.lower(),))
                agent_thread.start()
                agent_thread.join()
                end_confirm = Thread(target=pa.speak, args=['is there anything else i can do for you?'])
                end_confirm.start()
                end_confirm.join()
                text = pa.get_audio()
                if text.count('no') > 0  or text.count('stop') > 0 :
                    wake = 'stop emotion detection'
                    break
        elif text == 'stop':
            wake = ''
            speak_thread = Thread(target=pa.speak, args=['Good bye.'])
            speak_thread.start()
            speak_thread.join()
            break
