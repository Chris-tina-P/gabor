import math
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy.fft import rfft

from src.transformations.sound_transform import SoundTransform


class Fourier(SoundTransform):

    def read_wav(self, name):
        super().read_wav(name)
        self.frequencies = np.fft.rfftfreq(self.data_size, d=1. / self.samplerate)
        self.test = 0

    @abstractmethod
    def transform(self) -> (ndarray, ndarray):
        pass

    @abstractmethod
    def plot(self, fourier_data, frequencies, x_lim: int = 1000) -> None:
        pass


class OwnFourier(Fourier):
    def transform(self, data_part=None) -> (ndarray, ndarray):
        """
        This method computes and plots the Fourier transform of the loaded sound file with own fft.
        BUT DIFFERENT RESULTS THAN WITH SCIPY FFT!
        :return: None
        """
        # if data_part is None:
        #     data_part = self.data

        # fourier transform
        fourier_data = abs(self.fft(self.data))

        frequencies = np.fft.fftfreq(self.data_size, d=1. / self.samplerate)

        # plotting spectral content of sound wave
        # self.plot(fourier_data, frequencies)

        return fourier_data, frequencies

    def slow_dft(self, data) -> ndarray:
        """
        Compute the discrete Fourier Transform of the 1D array x
        :param data: The data to transform
        :return: The transformed data
        """
        x = np.asarray(data, dtype=float)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)

    def preprocess_fft(self, data) -> ndarray:
        """
        This method preprocesses the data for the FFT by adding zeros to the end of the data array.
        :param data: The data to preprocess
        :return: The preprocessed data, ready for the FFT
        """
        logarithm = int(math.log(len(data), 2))
        next_power_of_two = 2 ** (logarithm + 1)

        if len(data) < next_power_of_two:
            data = np.append(data, np.zeros(next_power_of_two - len(data)))

        return data

    def recursive_fft(self, data) -> ndarray:
        """
        This method computes the FFT of the data recursively.
        :param data: The data to transform
        :return: The transformed data
        """
        x = np.asarray(data, dtype=float)
        n = x.shape[0]

        cutoff = 32  # cutoff should be optimized
        if n <= cutoff:
            return self.slow_dft(x)
        else:
            x_even = self.recursive_fft(x[::2])
            x_odd = self.recursive_fft(x[1::2])
            factor = np.exp(-2j * np.pi * np.arange(n) / n)
            return np.concatenate([x_even + factor[:n // 2] * x_odd,
                                   x_even + factor[n // 2:] * x_odd])

    def fft(self, data) -> ndarray:
        """
        A recursive implementation of the 1D Cooley-Tukey FFT
        :param data: The data to transform
        :return: The transformed data
        """
        n_original = len(data)
        x = self.preprocess_fft(data)
        processed = self.recursive_fft(x)
        return processed[:n_original]

    def plot(self, fourier_data, frequencies, x_lim: int = 1000) -> None:
        """
        This method plots the Fourier transform of the transformed data.
        :param x_lim: The frequency limit of the plot (x limit)
        :param frequencies: The frequencies of the transformed data.
        :param fourier_data: The transformed data to plot.
        :return: None
        """
        plt.xlim([0, x_lim])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.plot(frequencies, fourier_data)
        plt.show()


class NpFourier(Fourier):

    def transform(self, data_part=None) -> (ndarray, ndarray):
        """
        This method computes and plots the Fourier transform of the loaded sound file with predefined fft.
        :return: None
        """
        if data_part is None:
            data_part = self.data

        # fourier transform
        fourier_data = np.abs(rfft(data_part))
        return fourier_data, self.frequencies

    def plot(self, fourier_data, frequencies, x_lim: int = 1000) -> None:
        """
        This method plots the Fourier transform of the transformed data.
        :param x_lim: The frequency limit of the plot (x limit)
        :param frequencies: The frequencies of the transformed data.
        :param fourier_data: The transformed data to plot.
        :return: None
        """
        plt.xlim([0, x_lim])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.plot(frequencies, fourier_data)
        plt.show()

