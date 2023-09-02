from abc import abstractmethod
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy.fft import rfft

from src.transformations.fourier import OwnFourier
from src.transformations.sound_transform import SoundTransform


def gaussian(x, mu, sig):
    return (
            1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

class Gabor(SoundTransform):
    def __init__(self, own_fourier: bool = False):
        super().__init__()
        self.data = None
        self.samplerate = None
        self.data_size = None
        self.song_length_seconds = None
        self.frequencies = None

        if own_fourier:
            self.fourier = OwnFourier().fft
        else:
            self.fourier = rfft

    def read_wav(self, name):
        super().read_wav(name)
        self.frequencies = np.fft.rfftfreq(self.data_size, d=1. / self.samplerate)

    @abstractmethod
    def transform(self, mean_data_values=5001, NFFT=10000, noverlap=500) -> (List[ndarray], ndarray, ndarray):
        pass

    @abstractmethod
    def plot(self, spectrum, frequencies, t, y_lim: int = 1000) -> None:
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

    def __int__(self, own_fourier: bool = False):
        super().__init__(own_fourier=own_fourier)

    # TODO: Add params to method signature
    def transform(self, mean_data_values=5001, NFFT=10000, noverlap=500) -> (List[ndarray], ndarray, ndarray):
        """
        This method computes und plots several fourier transforms of the loaded sound file
        TODO: plot spectrogram out of the fourier transforms

        """
        fft_to_use = self.fourier

        spectrum = []
        Fs = self.samplerate
        div = len(self.frequencies) // mean_data_values
        upper_limit = int((self.data_size - NFFT / 2 + 1 + NFFT - noverlap) // (NFFT - noverlap))

        # Cut out all frequencies below 0 and corresponding fourier data
        # positive_indices = self.freq_domain >= 0
        # self.freq_domain = self.freq_domain[positive_indices]
        # self.data = self.data[positive_indices]
        # self.data_size = self.data.shape[0]

        # Calculate the spectrum
        t = np.linspace(0, self.song_length_seconds, self.data_size)
        for i in range(0, upper_limit):
            window = gaussian(t, i * (NFFT - noverlap) / Fs, 0.1)

            gaussian_filtered = self.data * window

            fourier_data = abs(fft_to_use(gaussian_filtered))

            # Calculate the mean
            mean_data = np.zeros(mean_data_values)
            for j in range(0, mean_data_values):
                mean_data[j] = np.mean(fourier_data[j * div:(j + 1) * div])

            spectrum.append(mean_data)

        # convert spectrum to ndarray
        spectrum = np.asarray(spectrum)
        spectrum = spectrum.transpose()

        # Calculate frequency with the mean
        # freq = np.linspace(0, np.max(self.freq_domain), mean_data_values)
        freq = np.zeros(mean_data_values)
        for i in range(0, mean_data_values):
            freq[i] = np.mean(self.frequencies[i * div:(i + 1) * div])

        # Calculate time points
        t = np.arange(NFFT / 2, self.data_size - NFFT / 2 + 1, NFFT - noverlap) / Fs
        # np.linspace(0, self.song_length_seconds, 48)

        return spectrum, freq, t

    def plot(self, spectrum, frequencies, t, y_lim: int = 1000) -> None:
        plt.pcolormesh(t, frequencies, spectrum, shading='nearest', cmap='jet_r',
                       norm='log')  # shading='gouraud' makes diagram smoother
        plt.ylim([0, y_lim])
        plt.ylabel('Frequenz (Hz)')
        plt.xlabel('Zeit (s)')
        plt.title('Spektrogramm aus FFT-Werten')
        plt.colorbar(label='Leistungspegel (dB)')
        plt.show()

class NpGabor(Gabor):
    def transform(self, mean_data_values=5001, NFFT=10000, noverlap=500) -> (List[ndarray], ndarray, ndarray):
        spectrum, freqs, t, _ = plt.specgram(self.data, NFFT=NFFT, Fs=self.samplerate, noverlap=noverlap,
                                             cmap='jet_r')
        return spectrum, freqs, t

    def plot(self, spectrum, frequencies, t, y_lim: int = 1000) -> None:
        plt.pcolormesh(t, frequencies, spectrum, cmap='jet_r',
                       norm='log')  # shading='gouraud' makes diagram smoother
        plt.ylim([0, y_lim])
        plt.ylabel('Frequenz (Hz)')
        plt.xlabel('Zeit (s)')
        plt.title('Spektrogramm aus FFT-Werten')
        plt.colorbar(label='Leistungspegel (dB)')
        plt.show()

    def specgram(self, nfft: int = 10000, noverlap: int = 500, x_lim: int = 0, y_lim: int = 1000) -> None:
        """
        This method plots the spectrogram of the loaded sound file.
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
        plt.colorbar()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")
        plt.show()
