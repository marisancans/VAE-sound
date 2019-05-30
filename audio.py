#conda install -c anaconda pyaudio 
import pyaudio, wave, sys
import numpy as np

class AudioFile:
    chunk = 1024

    def __init__(self, file):
        """ Init audio stream """ 
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def play(self):
        """ Play entire file """
        frames = self.wf.readframes(self.chunk)
        while frames != '':
            amp = np.fromstring(frames, np.int16)
            amp = amp / 32768.0
            print(np.average(amp))
            self.stream.write(frames)
            frames = self.wf.readframes(self.chunk)

    def close(self):
        """ Graceful shutdown """ 
        self.stream.close()
        self.p.terminate()

# Usage example for pyaudio
a = AudioFile("taunt.wav")
a.play()
a.close()