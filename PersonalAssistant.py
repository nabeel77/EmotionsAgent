'''
Personal assistant is being implemented in this file
'''
import re
import os
import ray
import time
import pytz
import queue
import pickle
import random
import os.path
import pyttsx3
import requests
import datetime
import pyaudio
import wave
import threading
import speech_recognition as sr
from tempfile import TemporaryFile
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from selenium import webdriver
from youtubesearchpython import VideosSearch
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
          'december']
DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
DAY_EXTENSIONS = ['rd', 'th', 'st', 'nd']
SAD_MOOD_PHRASES = ['you look sad', 'it looks like you are in a sad mood', 'hey, u are sad']
SAD_MOOD_ACTIONS = ['would you like to hear a joke', 'would you like to watch something on youtube?']
POSITIVE_RESPONSES = ['yes', 'yes please', 'yes i would like that', 'that would be great', 'i like that']
NEGATIVE_RESPONSES = ['no', 'no i don\'t want that', 'not really']
youtube_str = 'youtube'
joke_str = 'joke'


# def playMp3(filename):
#     mp3file = AudioSegment.from_mp3(filename)
#     play(mp3file)
#
#
# def playWav(filename):
#     wavfile = AudioSegment.from_file(filename)
#     play(wavfile)


# If modifying these scopes, delete the file token.pickle.
def authenticate_google():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
            # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('calendar', 'v3', credentials=creds)
    return service


def get_events(day, service):
    # Call the Calendar API
    date = datetime.datetime.combine(day, datetime.datetime.min.time())
    end_date = datetime.datetime.combine(day, datetime.datetime.max.time())
    utc = pytz.UTC
    date = date.astimezone(utc)
    end_date = date.astimezone(utc)
    events_result = service.events().list(calendarId='primary', timeMin=date.isoformat(),
                                          timeMax=end_date.isoformat(), singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])

    if not events:
        speak('No events found for the required date')
        # print('No upcoming events found.')
    else:
        speak(f"you have {len(events)} events on this day.")
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        # print(start, event['summary'])
        start_time = str(start.split('T')[1].split('-')[0])
        if int(start_time.split(':')[0]) < 12:
            start_time = start_time + 'am'
        else:
            start_time = str(int(start_time.split(':')[0]) - 12)
            start_time = start_time + 'pm'
        speak(event["summary"] + " at " + start_time)


def get_date(text):
    text = text.lower()
    today = datetime.date.today()

    if text.count("today") > 0:
        return today

    day = -1
    day_of_week = -1
    month = -1
    year = today.year

    for word in text.split():
        if word in MONTHS:
            month = MONTHS.index(word) + 1
        elif word in DAYS:
            day_of_week = DAYS.index(word)
        elif word.isdigit():
            day = int(word)
        else:
            for ext in DAY_EXTENSIONS:
                found = word.find(ext)
                if found > 0:
                    try:
                        day = int(word[:found])
                    except:
                        pass

    if month < today.month and month != -1:
        year = year + 1

    if month == -1 and day != -1:  # if no month is found but a day is found
        if day < today.day:
            month = today.month + 1
        else:
            month = today.month

    if month == -1 and day == -1 and day_of_week != -1:
        current_day_of_week = today.weekday()
        dif = day_of_week - current_day_of_week

        if dif < 0:
            dif += 7
            if text.count("next") >= 1:
                dif += 7

        return today + datetime.timedelta(dif)

    if day != -1:
        return datetime.date(month=month, day=day, year=year)


SERVICE = authenticate_google()


def speak(sentence):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty("rate", 140)
    engine.say(sentence)
    engine.runAndWait()


def get_audio(queue=queue.Queue()):
    r = sr.Recognizer()
    m = None
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        if index == 1:
            m = sr.Microphone(device_index=index)
            break
    with sr.Microphone() as source:
        print('say something')
        audio = r.listen(source, timeout=60)
        r.adjust_for_ambient_noise(source)
        said = ''
        try:
            print('trying')
            said = r.recognize_google(audio)
            print(said)
        except sr.WaitTimeoutError as e:
            print("Timeout try again: {0}".format(e))
        except sr.UnknownValueError as e:
            print('Could not understand. Try again: ' + str(e))
        except Exception as e:
            print('Did not hear anything ' + str(e))
        queue.put(said.lower())
    return said.lower()


def joke_generator():
    joke_category_list = ['cat', 'dog', 'wife', 'husband', 'car', 'bike', 'friend', 'ghost', 'kid', 'time', 'love',
                          'fruit', 'furniture', 'sky diving', 'book', 'boy', 'girl']
    joke_item = random.choice(joke_category_list)
    information = requests.get(f"https://icanhazdadjoke.com/search?term={joke_item}",
                               headers={"Accept": "application/json"})
    connection = information.ok
    result = information.json()
    l_no_of_jokes = result["results"]
    no_of_jokes = len(l_no_of_jokes)
    select_joke = random.randrange(0, no_of_jokes)
    return l_no_of_jokes[select_joke]['joke']


def youtube_opener(id, stop):
    chrome_driver = 'chromedriver.exe'
    browser = webdriver.Chrome(chrome_driver)
    browser.get("https://www.youtube.com/watch?v=" + id)
    for i in range(5):
        time.sleep(1)
    length_str = browser.find_element_by_class_name("ytp-time-duration").text
    print(length_str)
    # browser.switch_to.frame(0)
    # WebDriverWait(browser, 10).until(EC.element_to_be_clickable((By.XPATH,"/html/body/div/c-wiz/div[2]/div/div/div/div/div[2]/form/div/span/span"))).click()
    try:
        WebDriverWait(browser, 5).until(EC.element_to_be_clickable((By.XPATH,
                                                                    '/html/body/ytd-app/ytd-popup-container/paper-dialog/yt-upsell-dialog-renderer/div/div[3]/div[1]/yt-button-renderer/a/paper-button/yt-formatted-string'))).click()
        browser._switch_to.frame("iframe")
        WebDriverWait(browser, 20).until(EC.element_to_be_clickable(
            (By.XPATH, "/html/body/div/c-wiz/div[2]/div/div/div/div/div[2]/form/div/span/span"))).click()
    except Exception as e:
        print("failed to click popup {}".format(e))
    while True:
        if stop():
            browser.quit()
            break


def youtube_handler(userdemand):
    artists = ['ed sheeran', 'rachel platten', 'post malone', 'nicki minaj', 'maroon 5', 'cardi b', 'katy perry',
               'drake', 'ImagineDragons']
    geners = ['comedy', 'vlogs', 'gaming videos', 'educational videos', 'songs', 'motivational videos']
    random_user_demands = ['anything', 'whatever you like', 'something']
    random_song_demands = ['any song', 'some song', 'a song', 'any music', 'play some music', 'play any song']
    song_geners = ['rock music', 'pop music', 'jazz music', 'popular music', 'country music', 'heavy metal music',
                   'folk music', 'hip hop music', 'sad music', 'celebration music', 'motivation music']
    stopping_commands = ['stop', 'turn off', 'shutdown', 'close', 'stop youtube', 'close youtube', 'stop the video',
                         'close the video', 'turn off video']

    if userdemand in random_user_demands:
        searchquery = random.choice(geners)
    elif userdemand == 'funny':
        searchquery = 'comedy'
    elif userdemand == 'play some music on youtube':
        searchquery = random.choice(song_geners)
    elif userdemand in random_song_demands:
        searchquery = random.choice(artists)
    else:
        searchquery = userdemand.replace('play ', '').replace(' by on youtube', '')
    print('playing: ', searchquery)
    videos_search = VideosSearch(searchquery, limit=5)
    id = videos_search.result()['result'][random.randrange(5)]['id']
    print(id)
    print(videos_search.result())
    speak('playing ' + searchquery + ' on youtube')
    stop_threads = False
    youtube_thread = threading.Thread(target=youtube_opener, args=(id, lambda: stop_threads,))
    youtube_thread.start()
    time.sleep(4)
    print('Say stop to stop youtube')
    que = queue.Queue()
    while True:
        command_thread = threading.Thread(target=get_audio, args=(que,))
        command_thread.start()
        command = que.get()
        if command != '' and command in stopping_commands:
            stop_threads = True
            youtube_thread.join()
            break


def command_checker(command):
    if command != '' and (command in POSITIVE_RESPONSES or command in NEGATIVE_RESPONSES):
        return command
    else:
        speak('did not understand what you want. try again')
        new_command = get_audio()
        while new_command == '' or new_command not in POSITIVE_RESPONSES or new_command not in NEGATIVE_RESPONSES:
            speak('did not understand you. try speaking more clearly')
            new_command = get_audio()
            if new_command in POSITIVE_RESPONSES or new_command in NEGATIVE_RESPONSES:
                return new_command
        return new_command


def decision(mood):
    if mood == 'sad':
        speak(random.choice(SAD_MOOD_PHRASES))
        action = random.choice(SAD_MOOD_ACTIONS)
        i = SAD_MOOD_ACTIONS.index(action)
        speak(action)
        command = get_audio()
        verified_command = command_checker(command)
        if verified_command in POSITIVE_RESPONSES:
            if i == 0:
                joke = joke_generator()
                print(joke)
                speak(joke)
            elif i == 1:
                speak('what would you like to watch?')
                command = get_audio()
                youtube_handler(command)
        elif verified_command in NEGATIVE_RESPONSES:
            speak('what would you like to do?')
            command = get_audio()
            if youtube_str in command:
                youtube_handler(command)
            elif joke_str in command:
                joke = joke_generator()
                print(joke)
                speak(joke)


if __name__ == '__main__':
    # testing get events function
    decision('sad')
