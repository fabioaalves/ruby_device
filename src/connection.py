import queue
import threading
import time
import websocket
import pyaudio


class Connection(threading.Thread):
    def __init__(self, url):
        self.socket = None
        self.stream = None
        self.url = url
        self.audio = pyaudio.PyAudio()
        threading.Thread.__init__(self)
        self.queue = None
        self.prepare_stream()

    def run(self):
        self.socket = websocket.WebSocketApp(self.url,
                                             on_message=self.on_message,
                                             on_error=self.on_error,
                                             on_close=self.on_close,
                                             on_open=self.on_open,
                                             header={"Device": "client"}
                                             )
        self.socket.run_forever(ping_interval=10)

    def is_connected(self):
        return self.socket.sock.connected

    def send(self, data):
        while not self.socket.sock.connected:
            time.sleep(0.25)

        print('Sending:', data)
        self.socket.send(data)

    def stop(self):
        print('Stopping the websocket...')
        self.socket.keep_running = False

    def on_message(self, ws, data):
        if isinstance(data, bytes):
            self.queue.put(data)
        else:
            print(data)

    def on_error(self, ws, error, a, b):
        print('Received error...')
        print(error)

    def on_close(self, ws):
        print('Closed the connection...')

    def on_open(self, ws):
        print('Opened the connection...')

    def play_audio(self):
        while True:
            audio_data = self.queue.get()
            if not audio_data:
                continue
            self.stream.write(audio_data)

    def prepare_stream(self):
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            output=True,
            frames_per_buffer=128
        )

        self.queue = queue.Queue()

        play_thread = threading.Thread(target=self.play_audio)
        play_thread.daemon = True
        play_thread.start()
