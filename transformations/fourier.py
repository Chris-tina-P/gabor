import math
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy.fft import rfft

from .sound_transform import SoundTransform


class Fourier(SoundTransform):
    """
    This class is used to compute and plot the Fourier transform of a sound file.
    """
    def read_wav(self, name: str) -> None:
        """
        Reads the wav file and computes the contained frequencies.
        :param name: name of the wav file
        :return: None
        """
        super().read_wav(name)
        self.frequencies = np.fft.rfftfreq(self.data_size, d=1. / self.samplerate)

    @abstractmethod
    def transform(self) -> (ndarray, ndarray):
        pass

    @abstractmethod
    def plot(self, fourier_data, freq, x_lim: int = 1000) -> None:
        pass

    def process(self, filename: str) -> None:
        """
        Loads the file, computes fourier transform and plots the result.
        :param filename: The name of the file.
        :return: None
        """
        super().process(filename)
        fourier_data, frequencies = self.transform()
        self.plot(fourier_data, frequencies)


class OwnFourier(Fourier):
    """
    This class is used to compute and plot the Fourier transform of a sound file with own methods.
    """
    def transform(self) -> (ndarray, ndarray):
        """
        This method computes and plots the Fourier transform of the loaded sound file with own fft.
        :return: The transformed data and the frequencies
        """
        fourier_data = abs(self.fft(self.data))
        frequencies = np.fft.fftfreq(len(fourier_data), d=1. / self.samplerate)

        # Remove negative frequencies
        fourier_data = fourier_data[:len(fourier_data)//2]
        frequencies = frequencies[:len(frequencies)//2]

        return fourier_data, frequencies

    def _slow_dft(self, data) -> ndarray:
        """
        Compute the discrete Fourier Transform of the 1D array data
        :param data: The data to transform
        :return: The transformed data
        """
        x = np.asarray(data, dtype=float)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)

    def _preprocess_fft(self, data) -> ndarray:
        """
        This method preprocesses the data for the FFT by adding zeros to the end of the data array.
        :param data: The data to preprocess
        :return: The preprocessed data ready for the FFT
        """
        logarithm = int(math.log(len(data), 2))
        next_power_of_two = 2 ** (logarithm + 1)

        if len(data) < next_power_of_two:
            data = np.append(data, np.zeros(next_power_of_two - len(data)))

        return data

    def _recursive_fft(self, data) -> ndarray:
        """
        This method computes the FFT of the data recursively.
        :param data: The data to transform
        :return: The transformed data
        """
        x = np.asarray(data, dtype=float)
        n = x.shape[0]

        cutoff = 32  # cutoff can be optimized
        if n <= cutoff:
            return self._slow_dft(x)
        else:
            x_even = self._recursive_fft(x[::2])
            x_odd = self._recursive_fft(x[1::2])
            factor = np.exp(-2j * np.pi * np.arange(n) / n)
            return np.concatenate([x_even + factor[:n // 2] * x_odd, x_even + factor[n // 2:] * x_odd])

    def fft(self, data) -> ndarray:
        """
        A recursive implementation of the 1D Cooley-Tukey FFT
        :param data: The data to transform
        :return: The transformed data
        """
        x = self._preprocess_fft(data)
        processed = self._recursive_fft(x)
        return processed

    def plot(self, fourier_data, freq, x_lim: int = 1000) -> None:
        """
        This method plots the Fourier transform of the transformed data.
        :param x_lim: The frequency limit of the plot (x limit)
        :param freq: The frequencies of the transformed data.
        :param fourier_data: The transformed data to plot.
        :return: None
        """
        plt.xlim([0, x_lim])
        plt.xlabel("Frequenz (Hz)")
        plt.ylabel("Amplitude")
        plt.plot(freq, fourier_data)
        plt.show()


class NpFourier(Fourier):
    """
    This class is used to compute and plot the Fourier transform of a sound file with numpy fft.
    """
    def transform(self) -> (ndarray, ndarray):
        """
        This method computes and plots the Fourier transform of the loaded sound file with predefined fft.
        :return: The transformed data and the frequencies
        """
        # fourier transform
        fourier_data = np.abs(rfft(self.data))
        return fourier_data, self.frequencies

    def plot(self, fourier_data, freq, x_lim: int = 1000) -> None:
        """
        This method plots the Fourier transformed data.
        :param x_lim: The frequency limit of the plot (x limit)
        :param freq: The frequencies of the transformed data.
        :param fourier_data: The transformed data to plot.
        :return: None
        """
        plt.xlim([0, x_lim])
        plt.xlabel("Frequenz (Hz)")
        plt.ylabel("Amplitude")
        plt.plot(freq, fourier_data)
        plt.show()
