import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Any, List

from src.transformations.fourier import OwnFourier, NpFourier
from src.transformations.gabor import OwnGabor, NpGabor


class TaskNotSupportedException(Exception):
    """
    Exception is thrown whenever a task is not supported.
    """


class Tasks(Enum):
    WAV = "WAV"
    FOURIER = "FOURIER"
    GABOR = "GABOR"
    SPECGRAM = "SPECGRAM"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        raise TaskNotSupportedException(f"{value} is a not supported Task!")

    def __str__(self) -> str:
        return self.value


def _set_up_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    sub_parser = arg_parser.add_subparsers(dest="command", required=True)

    wav_parser = sub_parser.add_parser(str(Tasks.WAV))
    fourier_parser = sub_parser.add_parser(str(Tasks.FOURIER))
    gabor_parser = sub_parser.add_parser(str(Tasks.GABOR))
    specgram_parser = sub_parser.add_parser(str(Tasks.GABOR))

    arg_parser.add_argument("input", type=Path, help="Path to the input file")

    fourier_parser.add_argument("--own", required=False, type=bool, default=False, help="Whether to use the own implementation or the library implementation")
    gabor_parser.add_argument("--own", required=False, type=bool, default=False, help="Whether to use the own implementation or the library implementation")

    fourier_parser.add_argument("--xlim", required=False, type=int, default=1000, help="The x limit for the plot")

    gabor_parser.add_argument("--num_data", required=False, type=int, default=5001, help="The number of data values to be averaged")
    gabor_parser.add_argument("--nfft", required=False, type=int, default=10000, help="The number of data points used in each block for the FFT")
    gabor_parser.add_argument("--noverlap", required=False, type=int, default=500, help="The number of points of overlap between blocks")
    gabor_parser.add_argument("--ylim", required=False, type=int, default=500, help="The y limit for the plot")

    specgram_parser.add_argument("--nfft", required=False, type=int, default=10000, help="The number of data points used in each block for the FFT")
    specgram_parser.add_argument("--noverlap", required=False, type=int, default=500, help="The number of points of overlap between blocks")
    specgram_parser.add_argument("--xlim", required=False, type=int, default=1000, help="The x limit for the plot")
    specgram_parser.add_argument("--ylim", required=False, type=int, default=500, help="The y limit for the plot")

    return arg_parser


def _run_wav(parsed_args):
    input_wav = parsed_args.input
    fourier = OwnFourier()
    fourier.read_wav(input_wav)
    fourier.plot_wav()


def _run_fourier(parsed_args):
    input_wav = parsed_args.input
    own = parsed_args.own
    xlim = parsed_args.xlim

    if own:
        fourier = OwnFourier()
    else:
        fourier = NpFourier()

    fourier.read_wav(input_wav)
    fourier_data, frequencies = fourier.transform()
    fourier.plot(fourier_data, frequencies, x_lim=xlim)


def _run_gabor(parsed_args):
    input_wav = parsed_args.input
    own = parsed_args.own
    num_data = parsed_args.num_data
    nfft = parsed_args.nfft
    noverlap = parsed_args.noverlap
    ylim = parsed_args.ylim

    if own:
        gabor = OwnGabor()
    else:
        gabor = NpGabor()

    gabor.read_wav(input_wav)
    spectrum, frequencies, t = gabor.transform(num_data=num_data, nfft=nfft, noverlap=noverlap)
    gabor.plot(spectrum, frequencies, t, y_lim=ylim)


def _run_specgram(parsed_args):
    input_wav = parsed_args.input
    nfft = parsed_args.nfft
    noverlap = parsed_args.noverlap
    xlim = parsed_args.xlim
    ylim = parsed_args.ylim

    gabor = NpGabor()

    gabor.read_wav(input_wav)
    gabor.specgram(nfft=nfft, noverlap=noverlap, x_lim=xlim, y_lim=ylim)


def main(args: List[str]) -> int:
    arg_parser = _set_up_arg_parser()
    parsed_args = arg_parser.parse_args(args)
    task = Tasks(parsed_args.command)

    if task == Tasks.WAV:
        _run_wav(parsed_args)
    elif task == Tasks.FOURIER:
        _run_fourier(parsed_args)
    elif task == Tasks.GABOR:
        _run_gabor(parsed_args)
    elif task == Tasks.SPECGRAM:
        _run_specgram(parsed_args)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
