import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile


class SoundTransform:
    """
    This class is used to read in a sound file and to plot the sound file as diagram with amplitude over time.
    """
    def __init__(self):
        self.data = None
        self.samplerate = None
        self.data_size = None
        self.song_length_seconds = None
        self.frequencies = None

    def read_wav(self, name: str):
        """
        This method reads in a sound file and prints the sound metadata.
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
        print("Datengröße:", self.data_size)
        print("Sample-Rate:", self.samplerate)
        print("Songlänge:", self.song_length_seconds, "s")

    def plot_wav(self) -> None:
        """
        This method plots the loaded sound file as diagram with amplitude over time.
        :return: None
        """
        # define time domain
        time = np.linspace(0, self.song_length_seconds, self.data_size)

        # plot sound file and add labels
        plt.plot(time, self.data)
        plt.xlabel("Zeit [s]")
        plt.ylabel("Amplitude")
        plt.show()

    def process(self, filename: str) -> None:
        """
        Loads the file, processes and plots the result.
        :param filename: The name of the file.
        :return: None
        """
        self.read_wav(filename)

