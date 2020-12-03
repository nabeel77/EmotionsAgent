from __future__ import print_function
import os
import ray
import time
import pickle
import os.path
import datetime
import pyttsx3
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
from tempfile import TemporaryFile
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import gtts_token as gt

SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
DAY_EXTENSIONS = ['rd', 'th', 'st']
# pyaudio.PyAudio().open(format=pyaudio.paInt16,
#                         rate=44100,
#                         channels=2, #change this to what your sound card supports
#                         input_device_index=0, #change this your input sound card index
#                         input=True,
#                         output=True,
#                         frames_per_buffer=512)

def playMp3(filename):
    mp3file = AudioSegment.from_mp3(filename)
    play(mp3file)


def playWav(filename):
    wavfile = AudioSegment.from_file(filename)
    play(wavfile)


def speak(sentence):
    tts = gTTS(text=sentence, lang='en')
    # # try:
    # #     tempAudio = TemporaryFile()
    filename = './new.mp3'
    tts.save(filename)
    # #    tempAudio.seek(0)
    playMp3(filename)
    # #    tempAudio.close()
    # # except Exception as e:
    # #     print('Exception arised: ' + str(e))
    # engine = pyttsx3.init()
    # engine.say(sentence)
    # engine.runAndWait()


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
    return said

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


def get_events(n, service):
    # Call the Calendar API
    now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    print(f'Getting the upcoming {n} events')
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                          maxResults=n, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])

    if not events:
        pass
        # print('No upcoming events found.')
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        # print(start, event['summary'])


def get_date(text):
    text = text.lower()
    today = datetime.date.today()
    if text.count('today') > 0:
        return today
    day = -1
    day_of_week = -1
    month = -1
    year = today.year
    for word in text.split():
        if word in MONTHS:
            month = MONTHS.index(word)+1
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
        year += 1
    if day < today.day and month == -1 and day != -1:
        month += 1
    if month == -1 and day == -1 and day_of_week != -1:
        current_day_of_week = today.weekday()
        diff = day_of_week - current_day_of_week
        if diff < 0:
            diff += 7
            if text.count('next') >= 1:
                diff += 7
        return today + datetime.timedelta(diff)
    return datetime.date(month = month, day = day, year = year)

# print(get_date(text))
if __name__ == '__main__':
    get_date('asd')
    text = get_audio().lower()
    # engine = pyttsx3.init()
    # engine.say("I will speak this text")
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voice.id)
    # engine.runAndWait()
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        engine.setProperty('voice', voice.id)  # changes the voice
        engine.say('The quick brown fox jumped over the lazy dog.')
    engine.runAndWait()
    #speak('Yo nigga')
    # text = get_audio()
    # if 'hello' in text:
    #     speak('hello, how are you?')
    # elif 'hi' in text:
    #     speak('hi, how are you?')
    # service = authenticate_google()
    # get_events(10, service)
