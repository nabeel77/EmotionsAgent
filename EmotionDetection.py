'''
Things done in this python file:
- Face and Emotion detection is done in this python file
Things to be done in this python file:
- Agent calls
'''

import ctypes
import cv2
import time
import argparse
import numpy as np
import TrainingModels as tm
from threading import Thread
import PersonalAssistant as pa
from keras.models import load_model
from multiprocessing import Process, Array, Value
from keras.preprocessing.image import img_to_array

height, width = 48, 48
batch_size = 64
validation_data_dir = './fer2013/validation'

classifier = load_model('./model3.h5')
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


# def face_detector(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, 1.3, 5)
#     if faces is ():
#         return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         roi_gray = gray[y:y + h, x:x + w]
#
#     try:
#         roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#     except:
#         return (x, w, y, h), np.zeros((48, 48), np.uint8), img
#     return (x, w, y, h), roi_gray, img
name = 'Hello Nabeel, how are you'


def startSoundThread():
    mus = Thread(target=pa.speak, args=[name])
    mus.start()


# def make_prediction():
#     cap = cv2.VideoCapture(0)
#     #confirmation = input('type yes if you want to detect the expressions: ')
#     counter = 1
#     while cap.isOpened():
#         while True:
#             ret, frame = cap.read()
#             rect, face, image = face_detector(frame)
#             emotion_detector(rect,face,image)
#             Show_cam(image)
#             if counter == 1:
#                 startSoundThread()
#                 #text = pa.get_audio()
#                 #print(text)
#                 counter = -1
#             # if counter == -1:
#             #     q = queue.Queue()
#             #     listener = Thread(target=pa.get_audio, args=(q,)).start()
#             #     result = q.get()
#             #     print(result)
#             #     counter = 2
#             #Show_cam(image)
#             if cv2.waitKey(1) == 13:  # 13 is the Enter Key
#                 break
#             # else:
#             #     Show_cam(image)
#             #     if cv2.waitKey(1) == 13:  # 13 is the Enter Key
#             #         break
#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         pa.speak('Camera is disconnected. Good bye.')


def emotion_detector(rect, face, image):
    label = ''
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the video image, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        #print(label)
        label_position = (rect[0] + int((rect[1] / 2)), rect[2] + 25)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(image, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    return label

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
            rect, face, image = face_detector_color(frame)
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
