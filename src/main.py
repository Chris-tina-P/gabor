from src.transformations.fourier import NpFourier, OwnFourier
from src.transformations.gabor import OwnGabor, NpGabor

# TODO: Remove commented out src that is unused
# TODO: Add and comment method parameters

# TODO: Rename class, maybe split into fourier and transformations class
# TODO: Remove unused methods

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

    ownGabor = OwnGabor()
    ownGabor.process('../input/Export1/Klavier_A_leicht.wav')

    npGabor = NpGabor()
    npGabor.process('../input/Export1/Klavier_A_leicht.wav')

    fourier = OwnFourier()
    fourier.process('../input/Export1/Klavier_A_leicht.wav')

    fourier = NpFourier()
    fourier.process('../input/Export1/Klavier_A_leicht.wav')