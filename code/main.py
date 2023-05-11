from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift
from IPython.display import clear_output


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
        # plot sound file
        plt.plot(self.data)
        plt.title("Sound file")
        plt.xlabel("Time")
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
        plt.xlim([-5000, 5000])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.plot(self.freq_domain, fourier_data_shift)
        plt.show()

    def gabor_transform2(self):
        time_domain = np.linspace(0, self.song_length_seconds, self.data_size)
        results = []

        for i in [2.2, 4.25, 6, 8, 10, 11.8]:
            clear_output(wait=True)
            plt.xlim([0, 600])
            gaussian = 11000 * np.exp(-2 * np.power(time_domain - i, 2))

            gaussian_filtered = self.data * gaussian

            fourier_data = abs(fft(gaussian_filtered))
            fourier_data_shift = fftshift(fourier_data)

            results.append(fourier_data_shift)

            plt.plot(self.freq_domain, fourier_data_shift)
            plt.pause(1)


if __name__ == '__main__':
    gabor = Gabor()
    gabor.read_wav('../input/1.wav')
    gabor.fourier_transform()
    gabor.gabor_transform2()