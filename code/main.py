import math

from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift
from IPython.display import clear_output
import stft


class Gabor:

    def __init__(self):
        self.data = None
        self.samplerate = None
        self.data_size = None
        self.song_length_seconds = None
        self.freq_domain = None

    def read_wav(self, name):
        # read in sound file
        self.samplerate, self.data = wavfile.read(name)

        # convert to mono
        if len(self.data.shape) > 1:
            self.data = self.data[:, 0]

        # define sound metadata
        self.data_size = self.data.shape[0]
        self.song_length_seconds = self.data_size / self.samplerate

        # define frequency domain
        self.freq_domain = np.linspace(- self.samplerate / 2, self.samplerate / 2, self.data_size)

        # print sound metadata
        print("Data size:", self.data_size)
        print("Sample rate:", self.samplerate)
        print("Song length (seconds):", self.song_length_seconds, "seconds")

    # This method plots the loaded sound file
    def plot_sound(self):
        # define time domain
        time = np.linspace(0, self.song_length_seconds, self.data_size)
        plt.plot(time, self.data)
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

    # This method computes the Gabor transform of the loaded sound file
    def gabor_transform(self):
        # define the window size and overlap
        window_size = 1024
        overlap = 0.5

        # compute the Gabor transform
        f, t, Sxx = signal.spectrogram(self.data, fs=self.samplerate, window='hann', detrend=False, scaling='spectrum')

        # plot the Gabor transform
        plt.specgram(self.data, Fs=self.samplerate, NFFT=window_size, noverlap=overlap * window_size, cmap='jet')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    def fourier_transform(self):

        # fourier transform
        fourier_data = abs(fft(self.data))
        fourier_data_shift = fftshift(fourier_data)

        # plotting spectral content of sound wave
        plt.xlim([0, 1000])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.plot(self.freq_domain, fourier_data_shift)
        plt.show()

    def fourier_transform2(self):
        # fourier transform
        fourier_data = abs(self.FFT(self.data))
        fourier_data_shift = self.fftshift(fourier_data)

        # plotting spectral content of sound wave
        plt.xlim([0, 5000])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.plot(self.freq_domain, fourier_data_shift)
        plt.show()

    def gabor_transform2(self):
        time_domain = np.linspace(0, self.song_length_seconds, self.data_size)
        results = []

        for i in [2.2, 4.25, 6, 8, 10, 11.8]:
            clear_output(wait=True)
            plt.xlim([0, 1000])
            gaussian = 11000 * np.exp(-2 * np.power(time_domain - i, 2))

            gaussian_filtered = self.data * gaussian

            fourier_data = abs(fft(gaussian_filtered))
            fourier_data_shift = fftshift(fourier_data)

            results.append(fourier_data_shift)

            plt.plot(self.freq_domain, fourier_data_shift)
            plt.pause(1)

    def gabor_transform3(self):
        plt.specgram(self.data, NFFT=128, Fs=10*self.samplerate, noverlap=120, cmap='jet_r')
        plt.ylim([0, 17000])
        plt.xlim([0, 2])
        plt.colorbar()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")
        plt.show()

    def DFT_slow(self, data):
        """Compute the discrete Fourier Transform of the 1D array x"""
        x = np.asarray(data, dtype=float)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)

    def preprocessFFT(self, data):
        logarithmus = int(math.log(len(data), 2))
        next_power_of_two = 2 ** (logarithmus + 1)

        if len(data) < next_power_of_two:
            data = np.append(data, np.zeros(next_power_of_two-len(data)))

        return data

    def recursiveFFT(self, data):
        x = np.asarray(data, dtype=float)
        N = x.shape[0]

        if N <= 32:  # this cutoff should be optimized
            return self.DFT_slow(x)
        else:
            X_even = self.recursiveFFT(x[::2])
            X_odd = self.recursiveFFT(x[1::2])
            factor = np.exp(-2j * np.pi * np.arange(N) / N)
            return np.concatenate([X_even + factor[:N // 2] * X_odd,
                                   X_even + factor[N // 2:] * X_odd])

    def FFT(self, data):
        """A recursive implementation of the 1D Cooley-Tukey FFT"""
        n_original = len(data)
        x = self.preprocessFFT(data)
        processed = self.recursiveFFT(x)
        return processed[:n_original]

    def fftshift(self, fourier_data):
        n = len(fourier_data)
        return np.concatenate((fourier_data[n // 2:], fourier_data[:n // 2]))


if __name__ == '__main__':
    gabor = Gabor()
    gabor.read_wav('../input/hbd.wav')
    # gabor.fourier_transform2()
    # gabor.read_wav('../input/hbd.wav')
    # gabor.fourier_transform2()
    gabor.gabor_transform3()


    # Interesting links
    # https://github.com/libAudioFlux/audioFlux
    # https://github.com/faroit/awesome-python-scientific-audio/blob/master/README.md
    # https://www.tutorialspoint.com/scipy/scipy_fftpack.htm
    # https://docs.scipy.org/doc/scipy/reference/fft.html
