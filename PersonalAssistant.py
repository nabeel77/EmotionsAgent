'''
Personal assistant is being implemented in this file
'''


import re
import time
import queue
import random
import pyttsx3
import requests
import datetime
import threading
from selenium import webdriver
import speech_recognition as sr
from bs4 import BeautifulSoup as bs
from selenium.webdriver.common.by import By
from youtubesearchpython import VideosSearch
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

Phrases_dict = {'sad': ['you look sad', 'it looks like you are in a sad mood', 'hey, u are sad'],
                'happy': ['good to see you happy', 'you look happy', 'seeing you happy is great'],
                'neutral': ['you look bored', 'being neutral can be boring']}
actions_dict = {'sad': ['would you like to hear a joke?', 'would you like to watch comedy on youtube?'],
                'happy': ['Here are some recommendations to celebrate happiness','would you like to watch educational videos on youtube?'],
                'neutral': ['would you like to play the number guessing game with me?', 'would you like to hear about interesting facts?']}
user_responses_dict = {'positive': ['yes', 'yes please', 'yes i would like that', 'that would be great', 'i like that'],
                       'negative': ['no', 'no i don\'t want that', 'not really']}
universal_command_dict = {'happiness_celebration': ['tell ways to celebrate happiness', 'how to celebrate happiness', 'what to do when i am happy'],
                          'interesting_facts': ['tell me interesting facts', 'what are some interesting facts', 'tell me some interesting facts', 'tell me interesting fact'],
                          'guess_game': ['let\'s play number guessing game', 'think of a number and i will guess']}
youtube_str = 'youtube'
joke_str = 'joke'
wished = 0


def speak(sentence):
    print(sentence)
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
            said = r.recognize_google(audio)
            print('you said: ', said)
        except sr.WaitTimeoutError as e:
            print("Timeout try again: {0}".format(e))
        except sr.UnknownValueError as e:
            pass
            # print('Could not understand. Try again: ' + str(e))
        except Exception as e:
            print('Did not hear anything ' + str(e))
        queue.put(said.lower())
    return said.lower()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good Morning Sir")

    elif 12 <= hour < 18:
        speak("Good Afternoon")

    else:
        speak("Good Evening")

    # assname = ("Jarvis 1 point o")
    speak("I am your Assistant")
    # speak(assname)


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


def fetch_happy_recommendations_from_internet():
    website_urls = [
        'https://www.businessinsider.com/science-backed-things-that-make-you-happier-2015-6#drink-coffee-not-too-much'
        '-though-3',
        'https://www.self.com/story/13-things-you-can-do-to-improve-your-mood-in-30-seconds-or-less']
    url_index = random.randint(0, 1)
    recommendations_list = []
    short_recommendations_list = []
    if url_index == 0:
        r = requests.get(website_urls[0])
        # fetching all html code from the website
        soup = bs(r.text, 'html.parser')
        # getting all recommendation heading tags form html code
        results = soup.find_all('h2', attrs={'class': 'slide-title-text'})
    else:
        r = requests.get(website_urls[1])
        soup = bs(r.text, 'html.parser')
        results = soup.find_all('div', attrs={'class': 'heading-h3'})
    # fetching plain text from html results
    for text in results:
        recommendations_list.append(text.string)
    rand_index = random.randrange(3, len(recommendations_list))
    for i in range(0, rand_index):
        rand_index = random.randrange(0, len(recommendations_list))
        if recommendations_list[rand_index] not in short_recommendations_list:
            short_recommendations_list.append(recommendations_list[rand_index])
        if i == 5:
            break
    for recommendation in short_recommendations_list:
        recommendation = re.sub("[^a-zA-Z ]+", "", recommendation).strip()
        speak(recommendation)


def guess_num():
    num = random.randint(40, 100)
    speak(f'I have generated a number from 1 to {num}. You have 8 tries to guess it. Good luck')
    num = str(random.randint(0, num))
    tries = 0
    maxTries = 8
    while True:
        while tries < maxTries:
            user_input = get_audio()
            if user_input > num:
                speak('your guessed number is larger than my number')
            elif user_input < num:
                speak('your guessed number is smaller than my number')
            else:
                speak('good job')
                break
            tries += 1
            if tries == maxTries:
                speak('you ran out of tries')
                break
        speak('want to play again?')
        confirm = get_audio()
        if confirm in user_responses_dict['negative']:
            break
        else:
            num = random.randint(40, 100)
            speak(f'I have generated a number from 1 to {num}. You have 8 tries to guess it. Good luck')
            num = str(random.randint(0, num))
            tries = 0


def fetch_interesting_facts_from_internet():
    # This function behaves the same way as fetch_happy_recommendations_from_internet().
    words_to_replace = ['we', 'We', 'our']
    replacement_words = ['Latvians', 'the']
    website_urls = [
        'https://bestlifeonline.com/world-facts/', 'https://www.whattoseeinlatvia.com/99-facts-about-latvia/']
    url_index = random.randint(0, 1)
    recommendations_list = []
    short_recommendations_list = []
    if url_index == 0:
        r = requests.get(website_urls[0])
        soup = bs(r.text, 'html.parser')
        results = soup.find_all('div', attrs={'class': 'title'})
        sentence = 'here are some interesting facts'
    else:
        r = requests.get(website_urls[1])
        soup = bs(r.text, 'html.parser')
        results = soup.find_all('p', attrs={'style': 'text-align: justify;'})
        sentence = 'here are some interesting facts about Latvia'
    speak(sentence)
    for text in results:
        recommendations_list.append(text.string)
    rand_index = random.randrange(3, len(recommendations_list))
    for i in range(0, rand_index):
        rand_index = random.randrange(0, len(recommendations_list))
        if recommendations_list[rand_index] not in short_recommendations_list:
            short_recommendations_list.append(recommendations_list[rand_index])
        if i == 5:
            break
    for recommendation in short_recommendations_list:
        if recommendation.startswith('#'):
            recommendation = recommendation.replace(recommendation[0:3], '').strip()
            recommendation = ' '.join([replacement_words[0] if word in words_to_replace else replacement_words[1] if
            word in words_to_replace else word for word in recommendation.split()])
        speak(recommendation)


def youtube_opener(id, stop):
    chrome_driver = 'chromedriver.exe'
    browser = webdriver.Chrome(chrome_driver)
    browser.get("https://www.youtube.com/watch?v=" + id)
    for i in range(5):
        time.sleep(1)
    length_str = browser.find_element_by_class_name("ytp-time-duration").text
    print(length_str)
    try:
        WebDriverWait(browser, 5).until(EC.element_to_be_clickable((By.XPATH,
                                                                    '/html/body/ytd-app/ytd-popup-container/paper'
                                                                    '-dialog/yt-upsell-dialog-renderer/div/div['
                                                                    '3]/div['
                                                                    '1]/yt-button-renderer/a/paper-button/yt'
                                                                    '-formatted-string'))).click()
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
    random_user_demands = ['anything', 'whatever you like', 'something']
    geners = ['comedy', 'vlogs', 'gaming videos', 'educational videos', 'songs', 'motivational videos']
    random_song_demands = ['any song', 'some song', 'a song', 'any music', 'play some music', 'play any song']
    artists = ['ed sheeran', 'rachel platten', 'post malone', 'nicki minaj', 'maroon 5', 'cardi b', 'katy perry', 'drake', 'ImagineDragons']
    stopping_commands = ['stop', 'turn off', 'shutdown', 'close', 'stop youtube', 'close youtube', 'stop the video', 'close the video', 'turn off video']
    song_geners = ['rock music', 'pop music', 'jazz music', 'popular music', 'country music', 'heavy metal music', 'folk music', 'hip hop music', 'sad music', 'celebration music', 'motivation music']

    if userdemand in random_user_demands:
        searchquery = random.choice(geners)
    elif userdemand == 'funny':
        searchquery = 'comedy'
    elif userdemand == 'play some music on youtube':
        searchquery = random.choice(song_geners)
    elif userdemand in random_song_demands:
        searchquery = random.choice(artists)
    elif userdemand == geners[3]:
        searchquery = geners[3]
    else:
        searchquery = userdemand.replace('play ', '').replace(' by on youtube', '')
    print('playing ', searchquery)
    videos_search = VideosSearch(searchquery, limit=5)
    video_id = videos_search.result()['result'][random.randrange(5)]['id']
    speak('playing ' + searchquery)
    stop_threads = False
    youtube_thread = threading.Thread(target=youtube_opener, args=(video_id, lambda: stop_threads,))
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
    if command != '' and (command in user_responses_dict['positive'] or command in user_responses_dict['negative']):
        return command
    else:
        speak('did not understand what you want. try again')
        new_command = get_audio()
        while new_command == '' or new_command not in user_responses_dict['positive'] or new_command not in \
                user_responses_dict['negative']:
            if new_command in user_responses_dict['positive'] or new_command in user_responses_dict['negative']:
                return new_command
            speak('did not understand you. try speaking more clearly')
            new_command = get_audio()

        return new_command


def universal_action():
    command = get_audio()
    if 'youtube' in command:
        youtube_handler(command)
    elif 'joke' in command:
        joke = joke_generator()
        print(joke)
        speak(joke)
    elif command in universal_command_dict['happiness_celebration']:
        speak('here are some ways to celebrate happiness')
        fetch_happy_recommendations_from_internet()
    elif command in universal_command_dict['interesting_facts']:
        fetch_interesting_facts_from_internet()
    elif command in universal_command_dict['guess_game']:
        speak('ok let\'s play')
        guess_num()
    else:
        speak('Sorry, I am not able to perform your desired action.')
        return


def decision(mood):
    global wished
    if wished == 0:
        wishMe()
        wished = 1
    if mood == 'sad':
        speak(random.choice(Phrases_dict['sad']))
        action = random.choice(actions_dict['sad'])
        i = actions_dict['sad'].index(action)
        speak(action)
        command = get_audio()
        verified_command = command_checker(command)
        if verified_command in user_responses_dict['positive']:
            if i == 0:
                joke = joke_generator()
                print(joke)
                speak(joke)
            else:
                command = 'funny'
                youtube_handler(command)
        elif verified_command in user_responses_dict['negative']:
            speak('what would you like to do?')
            universal_action()
    elif mood == 'happy':
        speak(random.choice(Phrases_dict['happy']))
        action = random.choice(actions_dict['happy'])
        i = actions_dict['happy'].index(action)
        print(i)
        speak(action)
        if i == 0:
            fetch_happy_recommendations_from_internet()
        else:
            command = get_audio()
            verified_command = command_checker(command)
            if verified_command in user_responses_dict['positive']:
                command = 'educational videos'
                youtube_handler(command)
            elif verified_command in user_responses_dict['negative']:
                speak('what would you like to do?')
                universal_action()
    elif mood == 'neutral':
        speak(random.choice(Phrases_dict['neutral']))
        action = random.choice(actions_dict['neutral'])
        i = actions_dict['neutral'].index(action)
        speak(action)
        command = get_audio()
        verified_command = command_checker(command)
        if verified_command in user_responses_dict['positive']:
            if i == 0:
                guess_num()
            else:
                fetch_interesting_facts_from_internet()
        elif verified_command in user_responses_dict['negative']:
            speak('what would you like to do?')
            universal_action()


if __name__ == '__main__':
    decision('neutral')
#   guess_num()
