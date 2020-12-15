'''
Personal assistant is being implemented in this file
'''

import os
import ray
import time
import pytz
import pickle
import os.path
import pyttsx3
import datetime
import threading
from gtts import gTTS
import gtts_token as gt
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
from tempfile import TemporaryFile
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow


SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
DAY_EXTENSIONS = ['rd', 'th', 'st', 'nd']

def playMp3(filename):
    mp3file = AudioSegment.from_mp3(filename)
    play(mp3file)


def playWav(filename):
    wavfile = AudioSegment.from_file(filename)
    play(wavfile)


def speak(sentence):
    # tts = gTTS(text=sentence, lang='en')
    # # try:
    # #     tempAudio = TemporaryFile()
    # filename = './new.mp3'
    # tts.save(filename)
    # #    tempAudio.seek(0)
    # playMp3(filename)
    # #    tempAudio.close()
    # # except Exception as e:
    # #     print('Exception arised: ' + str(e))
    print('name is: ', threading.current_thread().name)
    print(threading.get_ident())
    engine = pyttsx3.init()
    if sentence.lower() == 'sad':
        engine.say('why are you sad')
        engine.runAndWait()
    elif sentence.lower() == 'stop':
        engine.say('bye bro')
        engine.runAndWait()
    else:
        engine.say('Yo bro')
        engine.runAndWait()
    for t in threading.enumerate():
        print('active thread is: ', t.name)


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print('say something')
        audio = r.listen(source)
        said = ''
        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print('Exception: ' + str(e))
    return said.lower()

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
            for ext in DAY_EXTENTIONS:
                found = word.find(ext)
                if found > 0:
                    try:
                        day = int(word[:found])
                    except:
                        pass

    # THE NEW PART STARTS HERE
    if month < today.month and month != -1:  # if the month mentioned is before the current month set the year to the next
        year = year + 1

    # This is slighlty different from the video but the correct version
    if month == -1 and day != -1:  # if we didn't find a month, but we have a day
        if day < today.day:
            month = today.month + 1
        else:
            month = today.month

    # if we only found a dta of the week
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
if __name__ == '__main__':
    # testing get events function
    text = get_audio().lower()
# get_events(get_date(text), SERVICE)
# get_date('asd')
# text = get_audio().lower()
# engine = pyttsx3.init()
# engine.say("I will speak this text")
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voice.id)
# engine.runAndWait()
# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# for voice in voices:
#     engine.setProperty('voice', voice.id)  # changes the voice
#     engine.say('The quick brown fox jumped over the lazy dog.')
# engine.runAndWait()
# speak('Yo nigga')
# text = get_audio()
# if 'hello' in text:
#     speak('hello, how are you?')
# elif 'hi' in text:
#     speak('hi, how are you?')
# service = authenticate_google()
# get_events(10, service)
