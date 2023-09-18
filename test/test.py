from unittest import TestCase

from transformations.fourier import NpFourier, OwnFourier
from transformations.gabor import OwnGabor, NpGabor


class Test(TestCase):
    """
    This class is used to test the transformations.
    """

    # define input file
    input_file = '../input/Export1/Saw_A.wav'

    def test_plot_wav(self) -> None:
        """
        This method tests the plot_wav method.
        :return: None
        """
        own_fourier = OwnFourier()
        own_fourier.read_wav(self.input_file)
        own_fourier.plot_wav()

    def test_own_fourier(self) -> None:
        """
        This method tests the own fourier transformation.
        :return: None
        """
        own_fourier = OwnFourier()
        own_fourier.process('../input/Export1/Klavier_A_leicht.wav')
        # process is the same as read_wav, transform and plot
        own_fourier.read_wav(self.input_file)
        data, frequencies = own_fourier.transform()
        own_fourier.plot(data, frequencies)

    def test_np_fourier(self) -> None:
        """
        This method tests the numpy fourier transformation.
        :return: None
        """
        np_fourier = NpFourier()
        np_fourier.process(self.input_file)
        # process is the same as read_wav, transform and plot
        np_fourier.read_wav(self.input_file)
        data, frequencies = np_fourier.transform()
        np_fourier.plot(data, frequencies)

    def test_own_gabor(self) -> None:
        """
        This method tests the own gabor transformation.
        :return: None
        """
        own_gabor = OwnGabor()
        own_gabor.process(self.input_file)
        # process is the same as read_wav, transform and plot
        own_gabor.read_wav(self.input_file)
        spectrum, frequencies, t = own_gabor.transform()
        own_gabor.plot(spectrum, frequencies, t)

    def test_np_gabor(self) -> None:
        """
        This method tests the numpy gabor transformation.
        :return:
        """
        np_gabor = NpGabor()
        np_gabor.process(self.input_file)
        # process is the same as read_wav, transform and plot
        np_gabor.read_wav(self.input_file)
        spectrum, frequencies, t = np_gabor.transform()
        np_gabor.plot(spectrum, frequencies, t)
        # specgram is the numpy all in one implementation
        np_gabor.specgram()

