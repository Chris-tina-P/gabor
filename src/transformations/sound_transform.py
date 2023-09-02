import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile


class SoundTransform:

    def __init__(self):
        self.data = None
        self.samplerate = None
        self.data_size = None
        self.song_length_seconds = None
        self.frequencies = None

    def read_wav(self, name):
        """
        This method reads in a sound file and prints the sound metadata.
        TODO: Warning with own wav files
        :param name: The name of the sound file.
        :return: None
        """
        # read in sound file
        self.samplerate, self.data = wavfile.read(name)

        # convert to mono
        if len(self.data.shape) > 1:
            self.data = self.data[:, 0]

        # define sound metadata
        self.data_size = self.data.shape[0]
        self.song_length_seconds = self.data_size / self.samplerate

        # print sound metadata
        print("Data size:", self.data_size)
        print("Sample rate:", self.samplerate)
        print("Song length (seconds):", self.song_length_seconds, "seconds")

    def plot_wav(self) -> None:
        """
        This method plots the loaded sound file as diagram with amplitude over time.
        :return: None
        """
        # define time domain
        time = np.linspace(0, self.song_length_seconds, self.data_size)

        # plot sound file and add labels
        plt.plot(time, self.data)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

