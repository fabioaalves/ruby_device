from src.connection import Connection
from src.voice_recognition import VoiceRecognition
import asyncio


async def main():
    socket = Connection("ws://10.0.30.88:35500")
    socket.start()

    vr = VoiceRecognition(socket)
    vr.start_recognition()

asyncio.run(main())


