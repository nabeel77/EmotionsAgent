# from pydub import AudioSegment
# from pydub.playback import play
# import pyaudio
# import sounddevice as sd
#
# pa = pyaudio.PyAudio()
# print(pa.get_default_input_device_info())
# print(pyaudio.pa.__file__)
# print(pa.get_default_output_device_info())
#
#
# samplerates = 32000, 44100.0, 48000, 96000, 128000
# device = 0
#
# supported_samplerates = []
# for fs in samplerates:
#     try:
#         sd.check_output_settings(device=device, samplerate=fs)
#     except Exception as e:
#         print(fs, e)
#     else:
#         supported_samplerates.append(fs)
# print(supported_samplerates)
#
# pa = pyaudio.PyAudio()
# for x in range(0,pa.get_device_count()):
#     print (pa.get_device_info_by_index(x))
#
# pa = pyaudio.PyAudio()
# print(pa.get_default_host_api_info())
# # import pyglet
# #
# # song = pyglet.media.load('new.mp3')
# # song.play()
# # pyglet.app.run()
# # pyglet.app.exit()
#
#
#

lst = 'ML Data'
n = ' '
print(not n)