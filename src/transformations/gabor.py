from abc import abstractmethod
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy.fft import rfft
from scipy.fft import fft

from .fourier import OwnFourier
from .sound_transform import SoundTransform


class Gaussian:
    """
    This class is used to compute a gaussian function.
    """
    def __init__(self):
        pass

    @staticmethod
    def gaussian(x, mu, sig) -> float:
        """
        Gaussian function
        :param x: variable of the function
        :param mu: center of the peak
        :param sig: standard deviation
        :return: value of the function at x
        """
        return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)


class Gabor(SoundTransform):
    """
    This class is used to compute and plot the gabor transform of a sound file.
    """
    def __init__(self, own_fourier: bool = False):
        super().__init__()
        self.data = None
        self.samplerate = None
        self.data_size = None
        self.song_length_seconds = None
        self.frequencies = None

        if own_fourier:
            self.fourier = OwnFourier()._fft
        else:
            self.fourier = fft

    def read_wav(self, name: str) -> None:
        """
        Reads the wav file and computes the contained frequencies.
        :param name: name of the wav file
        :return: None
        """
        super().read_wav(name)
        self.frequencies = np.fft.rfftfreq(self.data_size, d=1. / self.samplerate)

    @abstractmethod
    def transform(self, num_data: int = 5001, nfft: int = 10000, noverlap: int = 500) \
            -> (List[ndarray], ndarray, ndarray):
        pass

    @abstractmethod
    def plot(self, spectrum: ndarray, frequencies: ndarray, t: ndarray, y_lim: int = 1000) -> None:
        pass

    def process(self, filename: str) -> None:
        """
        Loads the file, processes and plots the result.
        :param filename: The name of the file.
        :return: None
        """
        super().process(filename)
        spectrum, frequencies, t = self.transform()
        self.plot(spectrum, frequencies, t)


class OwnGabor(Gabor):
    """
    This class is used to compute and plot the gabor transform of a sound file with own methods.
    """
    def __int__(self, own_fourier: bool = False):
        super().__init__(own_fourier=own_fourier)

    def transform(self, num_data: int = 5001, nfft: int = 10000, noverlap: int = 500) -> (List[ndarray], ndarray, ndarray):
        """
        TODO: change gaussian window?
        This method computes several fourier transforms of the loaded sound file
        :param num_data: The number of data values to be averaged
        :param nfft: The number of data points used in each block for the FFT
        :param noverlap: The number of points of overlap between blocks
        :return: spectrum, frequencies and time points for plotting a spectrogram
        """
        fft_to_use = self.fourier

        spectrum = []
        Fs = self.samplerate
        div = len(self.frequencies) // num_data
        upper_limit = int((self.data_size - nfft / 2 + 1 + nfft - noverlap) // (nfft - noverlap))

        # Calculate the spectrum
        t = np.linspace(0, self.song_length_seconds, self.data_size)
        for i in range(0, upper_limit):
            window = Gaussian.gaussian(t, i * (nfft - noverlap) / Fs, 0.1)

            gaussian_filtered = self.data * window

            fourier_data = abs(fft_to_use(gaussian_filtered))

            # Calculate the mean
            mean_data = np.zeros(num_data)
            for j in range(0, num_data):
                mean_data[j] = np.mean(fourier_data[j * div:(j + 1) * div])

            spectrum.append(mean_data)

        # convert spectrum to ndarray
        spectrum = np.asarray(spectrum)
        spectrum = spectrum.transpose()

        # Calculate frequency with the mean
        freq = np.zeros(num_data)
        for i in range(0, num_data):
            freq[i] = np.mean(self.frequencies[i * div:(i + 1) * div])

        # Calculate time points
        t = np.arange(nfft / 2, self.data_size - nfft / 2 + 1, nfft - noverlap) / Fs

        return spectrum, freq, t

    def plot(self, spectrum: ndarray, frequencies: ndarray, t: ndarray, y_lim: int = 1000) -> None:
        """
        This method plots the spectrogram of the sound file.
        :param spectrum: color spectrum
        :param frequencies: y-axis
        :param t: x-axis
        :param y_lim: limit for the y-axis
        :return: None
        """
        # shading='gouraud' makes diagram smoother
        plt.pcolormesh(t, frequencies, spectrum, shading='nearest', cmap='jet_r', norm='log')
        plt.ylim([0, y_lim])
        plt.ylabel('Frequenz (Hz)')
        plt.xlabel('Zeit (s)')
        plt.show()


class NpGabor(Gabor):
    """
    This class is used to compute and plot the gabor transform of a sound file with given numpy functions.
    """
    def transform(self, num_data: int = 5001, nfft: int = 10000, noverlap: int = 500) -> (List[ndarray], ndarray, ndarray):
        """
        This method computes und plots several fourier transforms of the loaded sound file
        :param num_data: The number of data values to be averaged
        :param nfft: The number of data points used in each block for the FFT
        :param noverlap: The number of points of overlap between blocks
        :return: spectrum, frequencies and time points for plotting a spectrogram
        """
        spectrum, freqs, t, _ = plt.specgram(self.data, NFFT=nfft, Fs=self.samplerate, noverlap=noverlap,
                                             cmap='jet_r')
        return spectrum, freqs, t

    def plot(self, spectrum, frequencies, t, y_lim: int = 1000) -> None:
        """
        This method plots the spectrogram of the sound file.
        :param spectrum: color spectrum
        :param frequencies: y-axis
        :param t: x-axis
        :param y_lim: limit for the y-axis
        :return: None
        """
        # shading='gouraud' makes diagram smoother
        plt.pcolormesh(t, frequencies, spectrum, cmap='jet_r', norm='log')
        plt.ylim([0, y_lim])
        plt.ylabel('Frequenz (Hz)')
        plt.xlabel('Zeit (s)')
        plt.show()

    def specgram(self, nfft: int = 10000, noverlap: int = 500, x_lim: int = 0, y_lim: int = 1000) -> None:
        """
        This method computes and plots the spectrogram of the loaded sound file with given method specgram.
        :param nfft: The number of points for the fft.
        :param noverlap: The number of points for the overlap.
        :param x_lim: The time limit (x limit) of the plot. If smaller or equal 0, the whole song is plotted.
        :param y_lim: The frequency limit (y limit) of the plot.
        :return: None
        """
        if x_lim <= 0:
            x_lim = self.song_length_seconds

        plt.specgram(self.data, NFFT=nfft, Fs=self.samplerate, noverlap=noverlap, cmap='jet_r')
        plt.ylim([0, y_lim])
        plt.xlim([0, x_lim])
        plt.xlabel("Zeit (s)")
        plt.ylabel("Frequenz (Hz)")
        plt.show()
