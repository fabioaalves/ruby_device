from src.connection import Connection
from src.voice_recognition import VoiceRecognition
import threading
import asyncio

socket = None


def start_connection():
    global socket
    socket = Connection("ws://10.0.30.88:35500")
    socket.start()


async def main():
    global socket
    play_thread = threading.Thread(target=start_connection())
    play_thread.daemon = True
    play_thread.start()

    vr = VoiceRecognition(socket)
    vr.start_recognition()

asyncio.run(main())
