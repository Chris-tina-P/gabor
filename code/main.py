import math
from abc import abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from numpy.fft import rfft
from scipy.io import wavfile


def gaussian(x, mu, sig):
    return (
            1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


# TODO: Remove commented out code that is unused
# TODO: Add and comment method parameters

# TODO: Rename class, maybe split into fourier and gabor class
# TODO: Remove unused methods

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

    # @abstractmethod
    # def transform(self):
    #     pass
    #
    # @abstractmethod
    # def plot(self, fourier_data, freq_domain,  x_lim: int = 1000) -> None:
    #     pass


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
        spectrum, freqs, t, _ = plt.specgram(self.data, NFFT=NFFT, Fs=self.samplerate, noverlap=noverlap, cmap='jet_r')
        return spectrum, freqs, t

    def plot(self, spectrum, frequencies, t, y_lim: int = 1000) -> None:
        plt.pcolormesh(t, frequencies, spectrum, cmap='jet_r', norm='log')  # shading='gouraud' makes diagram smoother
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
        self.plot_fourier(fourier_data, self.freq_domain)

    def plot_fourier(self, fourier_data, freq_domain, x_lim: int = 1000) -> None:
        """
        This method plots the Fourier transform of the transformed data.
        :param x_lim: The frequency limit of the plot (x limit)
        :param fourier_data: The transformed data to plot.
        :return: None
        """
        plt.xlim([0, x_lim])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.plot(freq_domain, fourier_data)
        plt.show()

    def own_fourier_transform(self) -> None:
        """
        This method computes and plots the Fourier transform of the loaded sound file with own fft.
        BUT DIFFERENT RESULTS THAN WITH SCIPY FFT!
        :return: None
        """
        # fourier transform
        fourier_data = abs(self.fft(self.data))

        freq_domain = np.fft.fftfreq(self.data_size, d=1. / self.samplerate)

        # plotting spectral content of sound wave
        self.plot_fourier(fourier_data, freq_domain)

    # TODO: Add params to method signature
    def windowed_fourier_transform(self, x_lim: int = 1000, own_fourier: bool = False):
        """
        This method computes und plots several fourier transforms of the loaded sound file
        TODO: plot spectrogram out of the fourier transforms

        """
        fft_to_use = rfft
        if own_fourier:
            fft_to_use = self.fft

        spectrum = []
        mean_data_values = 5001
        NFFT = 10000
        noverlap = 500
        Fs = self.samplerate
        div = len(self.freq_domain) // mean_data_values
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
            freq[i] = np.mean(self.freq_domain[i * div:(i + 1) * div])

        # Calculate time points
        t = np.arange(NFFT / 2, self.data_size - NFFT / 2 + 1, NFFT - noverlap) / Fs
        # np.linspace(0, self.song_length_seconds, 48)

        return spectrum, freq, t

    def own_gabor_transform(self, y_lim=2000, x_lim=0, own_fourier=False) -> None:
        """
        This method computes the Gabor transform of the loaded sound file. It uses the own windowed fourier transform.
        :return: None
        """
        spectrum, freqs, t = self.windowed_fourier_transform(own_fourier=own_fourier)

        # im = plt.imshow(spectrum, cmap='jet', vmin=0, vmax=y_lim, extent=[0, 1200, 0, 1100])
        # plt.colorbar(im)
        # plt.show()

        plt.pcolormesh(t, freqs, spectrum, shading='nearest', cmap='jet_r',
                       norm='log')  # shading='gouraud' makes diagram smoother
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
    # fourier = OwnFourier()
    # fourier.read_wav('../input/Export1/Klavier_A_leicht.wav')
    # data, frequencies = fourier.transform()
    # fourier.plot(data, frequencies)

    # npGabor = NpGabor()
    # npGabor.read_wav('../input/Export1/Klavier_A_leicht.wav')
    # spectrum, frequencies, t = npGabor.transform()
    # npGabor.plot(spectrum, frequencies, t)
    # npGabor.specgram()

    # ownGabor = OwnGabor()
    # ownGabor.read_wav('../input/Export1/Klavier_A_leicht.wav')
    # spectrum, frequencies, t = ownGabor.transform()
    # ownGabor.plot(spectrum, frequencies, t)

    ownGabor = OwnGabor(own_fourier=True)
    ownGabor.read_wav('../input/Export1/Klavier_A_leicht.wav')
    spectrum, frequencies, t = ownGabor.transform()
    ownGabor.plot(spectrum, frequencies, t)
