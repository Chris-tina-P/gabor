import math

from numpy import ndarray
from numpy.fft import rfft
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift
from IPython.display import clear_output
from typing import List


class Gabor:
    """
    This class is used to read in a sound file and compute the Gabor transform of the sound file.
    """

    def __init__(self):
        self.data = None
        self.samplerate = None
        self.data_size = None
        self.song_length_seconds = None
        self.freq_domain = None

    def read_wav(self, name) -> None:
        """
        This method reads in a sound file and prints the sound metadata.
        TODO: Error with own wav files
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

        self.freq_domain = np.fft.rfftfreq(self.data_size, d=1. / self.samplerate)

        # print sound metadata
        print("Data size:", self.data_size)
        print("Sample rate:", self.samplerate)
        print("Song length (seconds):", self.song_length_seconds, "seconds")

    def plot_sound(self) -> None:
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

    def fourier_transform(self) -> None:
        """
        This method computes and plots the Fourier transform of the loaded sound file with predefined fft.
        :return: None
        """
        # fourier transform
        fourier_data = np.abs(rfft(self.data))
        self.plot_fourier(fourier_data)

    def plot_fourier(self, fourier_data, x_lim: int = 1000) -> None:
        """
        This method plots the Fourier transform of the transformed data.
        :param x_lim: The frequency limit of the plot (x limit)
        :param fourier_data: The transformed data to plot.
        :return: None
        """
        plt.xlim([0, x_lim])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.plot(self.freq_domain, fourier_data)
        plt.show()

    def own_fourier_transform(self) -> None:
        """
        This method computes and plots the Fourier transform of the loaded sound file with own fft.
        BUT DIFFERENT RESULTS THAN WITH SCIPY FFT!
        :return: None
        """
        # fourier transform
        fourier_data = abs(self.fft(self.data))
        fourier_data_shift = self.fft_shift(fourier_data)

        # plotting spectral content of sound wave
        self.plot_fourier(fourier_data_shift)

    def windowed_fourier_transform(self, x_lim: int = 1000):
        """
        This method computes und plots several fourier transforms of the loaded sound file
        TODO: plot spectrogram out of the fourier transforms

        """
        spectrum = []
        mean_data_values = 5000
        div = len(self.freq_domain) // mean_data_values

        # Cut out all frequencies below 0 and corresponding fourier data
        # positive_indices = self.freq_domain >= 0
        # self.freq_domain = self.freq_domain[positive_indices]
        # self.data = self.data[positive_indices]
        # self.data_size = self.data.shape[0]

        # Calculate the spectrum
        t = np.linspace(0, self.song_length_seconds, self.data_size)
        # TODO: why 8 and 11000?
        for i in range(0, 48):
            gaussian = 11000 * np.exp(-2 * np.power(t - i/6, 2))

            gaussian_filtered = self.data * gaussian

            fourier_data = abs(rfft(gaussian_filtered))

            # Calculate the mean
            mean_data = np.zeros(mean_data_values)
            for j in range(0, mean_data_values):
                mean_data[j] = np.mean(fourier_data[j * div:(j + 1) * div])

            spectrum.append(mean_data)

        # convert spectrum to ndarray
        spectrum = np.asarray(spectrum)
        spectrum = spectrum.transpose()

        # Calculate frequency
        freq = np.linspace(0, np.max(self.freq_domain), mean_data_values)

        # Calculate time points
        NFFT = 10000
        noverlap = 500
        Fs = self.samplerate

        t = np.arange(NFFT/2, self.data_size - NFFT/2+1, NFFT - noverlap) / Fs

        return spectrum, freq, t

    def own_gabor_transform(self, y_lim=1000, x_lim=0) -> None:
        """
        This method computes the Gabor transform of the loaded sound file. It uses the own windowed fourier transform.
        :return: None
        """
        spectrum, freqs, t = self.windowed_fourier_transform()

        # im = plt.imshow(spectrum, cmap='jet', vmin=0, vmax=y_lim, extent=[0, 1200, 0, 1100])
        # plt.colorbar(im)
        # plt.show()

        plt.pcolormesh(t, freqs, spectrum, shading='nearest', cmap='jet_r', norm='log')  # shading='gouraud' makes diagram smoother
        plt.ylim([0, y_lim])
        plt.ylabel('Frequenz (Hz)')
        plt.xlabel('Zeit (s)')
        plt.title('Spektrogramm aus FFT-Werten')
        plt.colorbar(label='Leistungspegel (dB)')
        plt.show()

    def gabor_own_plot(self, nfft: int = 10000, noverlap: int = 500, x_lim: int = 0, y_lim: int = 1000) -> None:
        if x_lim <= 0:
            x_lim = self.song_length_seconds

        spectrum, freqs, t, _ = plt.specgram(self.data, NFFT=nfft, Fs=self.samplerate, noverlap=noverlap, cmap='jet_r')

        plt.pcolormesh(t, freqs, spectrum, cmap='jet_r', norm='log')  # shading='gouraud' makes diagram smoother
        plt.ylim([0, y_lim])
        plt.ylabel('Frequenz (Hz)')
        plt.xlabel('Zeit (s)')
        plt.title('Spektrogramm aus FFT-Werten')
        plt.colorbar(label='Leistungspegel (dB)')
        plt.show()

    def gabor_transform(self, nfft: int = 10000, noverlap: int = 500, x_lim: int = 0, y_lim: int = 1000) -> None:
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

    def fft_shift(self, fourier_data) -> ndarray:
        """
        This method shifts the fourier data to the left.
        :param fourier_data: The fourier data to shift
        :return: The shifted fourier data
        """
        n = len(fourier_data)
        return np.concatenate((fourier_data[n // 2:], fourier_data[:n // 2]))


if __name__ == '__main__':
    gabor = Gabor()
    gabor.read_wav('../input/Export1/Taka_a_E2_5.wav')
    # gabor.read_wav('../input/hbd.wav')
    # gabor.own_fourier_transform()
    # gabor.fourier_transform()
    # gabor.plot_sound()
    # gabor.gabor_transform()
    gabor.gabor_own_plot()
    gabor.own_gabor_transform()
    # gabor.gabor_transform()

    # Interesting links
    # https://github.com/libAudioFlux/audioFlux
    # https://github.com/faroit/awesome-python-scientific-audio/blob/master/README.md
    # https://www.tutorialspoint.com/scipy/scipy_fftpack.htm
    # https://docs.scipy.org/doc/scipy/reference/fft.html
