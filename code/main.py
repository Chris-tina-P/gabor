from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift


class Gabor:

    def __init__(self):
        self.data = None

    def read_wav(self, name):
        # read in sound file
        samplerate, self.data = wavfile.read(name)

        # define sound metadata
        data_size = self.data.shape[0]
        song_length_seconds = data_size / samplerate

        # print sound metadata
        print("Data size:", data_size)
        print("Sample rate:", samplerate)
        print("Song length (seconds):", song_length_seconds, "seconds")

    # This method plots the loaded sound file
    def plot_sound(self):
        # plot sound file
        plt.plot(self.data)
        plt.title("Sound file")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()


if __name__ == '__main__':
    gabor = Gabor()
    gabor.read_wav('../input/1.wav')
    gabor.plot_sound()
