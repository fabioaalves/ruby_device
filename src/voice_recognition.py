import os
import pyaudio
import wave
import numpy as np
import speech_recognition as sr
from datetime import datetime, timedelta
from lib.audio_processing import Resnet50_Arc_loss
from lib.engine import HotwordDetector
from lib.streams import SimpleMicStream


class VoiceRecognition:
    def __init__(self, sample_rate=44100, channels=2):
        self.chunk = 1024
        self.output_file = "audio.wav"
        self.duration = 5
        self.sample_rate = sample_rate
        self.channels = channels

        path = os.path.dirname(os.path.realpath(__file__))
        ref_file = os.listdir(os.path.join(path, "lib/ruby_ref.json"))

        self.base_model = Resnet50_Arc_loss()
        self.ruby = HotwordDetector(
            hotword="ruby",
            model=self.base_model,
            reference_file=ref_file,
            threshold=0.7,
            relaxation_time=2
        )

    def get_stream(self):
        return SimpleMicStream(
            window_length_secs=1.5,
            sliding_window_secs=0.4
        )

    def record_audio(self):
        peak_counter = 0
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16,
                            channels=self.channels,
                            rate=self.sample_rate,
                            input=True,
                            frames_per_buffer=self.chunk)

        frames = []
        recording = True
        start_time = datetime.now()
        peak_detected = False

        while recording:
            data = stream.read(self.chunk)
            frames.append(data)

            audio_data = np.frombuffer(data, dtype=np.int16)

            print(np.max(audio_data))
            if np.max(audio_data) > 200:
                peak_counter += 1

            if peak_counter > 5:
                peak_detected = True

            elapsed_time = datetime.now() - start_time
            if elapsed_time >= timedelta(seconds=self.duration) or peak_detected is True:
                if np.max(audio_data) > 40:
                    recording = True
                else:
                    recording = False

            if elapsed_time >= timedelta(seconds=45):
                recording = False

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(self.output_file, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))

        print("Gravação concluída e salva em", self.output_file)

    def recognize_speech(self):
        r = sr.Recognizer()
        with sr.AudioFile(self.output_file) as source:
            audio_data = r.record(source)

        try:
            text = r.recognize_google(audio_data)
            print("Recognized speech:", text)
        except sr.UnknownValueError:
            print("Could not understand speech")
        except sr.RequestError as e:
            print("Recognition request failed:", str(e))

    def start_recognition(self):
        stream = self.get_stream()
        stream.start_stream()
        print("Say Ruby")
        while True:
            frame = stream.getFrame()
            result = self.ruby.scoreFrame(frame)
            if result is None:
                continue
            if result["match"]:
                stream.close_stream()
                print("Wakeword uttered", result["confidence"])
                self.record_audio()
                self.recognize_speech()
                break
        self.start_recognition()
