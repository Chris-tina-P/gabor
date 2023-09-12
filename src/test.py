from transformations.fourier import NpFourier, OwnFourier
from transformations.gabor import OwnGabor, NpGabor

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

    # ownGabor = OwnGabor()
    # ownGabor.process('../input/Export1/Klavier_A_leicht.wav')

    # npGabor = NpGabor()
    # npGabor.process('../input/Export1/Klavier_A_leicht.wav')
    # npGabor.process('../input/Export1/Klavier_A_Stark.wav')

    # npGabor.read_wav('../input/Export1/Klavier_A_leicht.wav')
    # npGabor.specgram(y_lim=2000)
    #
    # npGabor.read_wav('../input/Export1/Klavier_A_Stark.wav')
    # npGabor.specgram()

    fourier = OwnFourier()
    # fourier.process('../input/Export1/Klavier_A_leicht.wav')

    np_fourier = NpFourier()
    # fourier.process('../input/Export1/Klavier_A_leicht.wav')

    # npGabor.read_wav('../input/Export1/Klavier_A-1.wav')
    # npGabor.specgram(x_lim=7)

    # fourier = NpFourier()
    # fourier.process('../input/Export1/Saw_A.wav')
    # fourier.process('../input/Export1/Sin_A.wav')

    # npGabor.read_wav('../input/Export1/Strat_a_mittel.wav')
    # npGabor.specgram(x_lim=20)
    #
    # npGabor.read_wav('../input/Export1/Tele_a_oben.wav')
    # npGabor.specgram(x_lim=20)
    #
    # npGabor.read_wav('../input/Export1/Strat_a_oben.wav')
    # npGabor.specgram(x_lim=20)
    #
    # npGabor.read_wav('../input/Export1/Tele_a_unten.wav')
    # npGabor.specgram(x_lim=20)
    #
    # fourier.process('../input/Export1/Tele_a_oben.wav')
    #
    # fourier.process('../input/Export1/Strat_a_oben.wav')
    #
    # fourier.process('../input/Export1/Tele_a_unten.wav')
    #
    # npGabor.read_wav('../input/hbd.wav')
    # npGabor.specgram()
    # npGabor.specgram(y_lim=550, x_lim=24)

    # own_own_gabor = OwnGabor(own_fourier=True)
    # own_gabor = OwnGabor()
    np_gabor = NpGabor()
    input_file = 'input/Export1/Saw_A.wav'

    # np_gabor.read_wav(input_file)
    # np_gabor.specgram()

    # np_gabor.process(input_file)

    # own_gabor.read_wav(input_file)
    # spectrum, freq, t = own_gabor.transform()
    # own_gabor.plot(spectrum, freq, t)
    #
    # own_own_gabor.process(input_file)

    # own_fourier = OwnFourier()
    # own_fourier.process('../input/Export1/Klavier_A_leicht.wav')

    fourier.read_wav(input_file)
    data, frequencies = fourier.transform()
    fourier.plot(data, frequencies, x_lim=3000)
    np_gabor.read_wav(input_file)
    np_gabor.specgram(x_lim=5, y_lim=3000)


