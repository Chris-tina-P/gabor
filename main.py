import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Any, List

from transformations.fourier import OwnFourier, NpFourier
from transformations.gabor import OwnGabor, NpGabor


class TaskNotSupportedException(Exception):
    """
    Exception is thrown whenever a task is not supported.
    """


class Tasks(Enum):
    """
    Enum for the different tasks that can be executed.
    """
    WAV = "WAV"
    FOURIER = "FOURIER"
    GABOR = "GABOR"
    SPECGRAM = "SPECGRAM"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        """
        Method is called whenever a task is not supported.
        :param value: The value that is not supported.
        :return: The exception that is thrown.
        """
        raise TaskNotSupportedException(f"{value} is a not supported Task!")

    def __str__(self) -> str:
        """
        Method is called whenever the enum is converted to a string.
        :return: The string representation of the enum.
        """
        return self.value


def _set_up_arg_parser() -> ArgumentParser:
    """
    Sets up the argument parser. The parser is used to parse the command line arguments.
    :return: The argument parser.
    """
    arg_parser = ArgumentParser()
    sub_parser = arg_parser.add_subparsers(dest="command", required=True)

    # Add sub parsers for the different tasks
    sub_parsers = []
    wav_parser = sub_parser.add_parser(str(Tasks.WAV))
    sub_parsers.append(wav_parser)
    fourier_parser = sub_parser.add_parser(str(Tasks.FOURIER))
    sub_parsers.append(fourier_parser)
    gabor_parser = sub_parser.add_parser(str(Tasks.GABOR))
    sub_parsers.append(gabor_parser)
    specgram_parser = sub_parser.add_parser(str(Tasks.SPECGRAM))
    sub_parsers.append(specgram_parser)

    # Add arguments for fourier task
    fourier_parser.add_argument("--own", action="store_true", required=False, default=False,
                                help="Whether to use the own implementation or the library implementation")
    fourier_parser.add_argument("--xlim", required=False, type=int, default=1000, help="The x limit for the plot")

    # Add arguments for gabor task
    gabor_parser.add_argument("--own", action="store_true", required=False, default=False,
                              help="Whether to use the own implementation or the library implementation")
    gabor_parser.add_argument("--num_data", required=False, type=int, default=5001,
                              help="The number of data values to be averaged")
    gabor_parser.add_argument("--nfft", required=False, type=int, default=10000,
                              help="The number of data points used in each block for the FFT")
    gabor_parser.add_argument("--noverlap", required=False, type=int, default=500,
                              help="The number of points of overlap between blocks")
    gabor_parser.add_argument("--ylim", required=False, type=int, default=500, help="The y limit for the plot")

    # Add arguments for specgram task
    specgram_parser.add_argument("--nfft", required=False, type=int, default=10000,
                                 help="The number of data points used in each block for the FFT")
    specgram_parser.add_argument("--noverlap", required=False, type=int, default=500,
                                 help="The number of points of overlap between blocks")
    specgram_parser.add_argument("--xlim", required=False, type=int, default=0, help="The x limit for the plot")
    specgram_parser.add_argument("--ylim", required=False, type=int, default=1000, help="The y limit for the plot")

    # Add arguments (input file) for all tasks
    for parser in sub_parsers:
        parser.add_argument("input", type=Path, help="Path to the input file")

    return arg_parser


def _run_wav(parsed_args: Any) -> None:
    """
    Runs the wav task. The wav task is used to plot the wav file.
    :param parsed_args: The parsed arguments.
    :return: None
    """
    # Get the arguments
    input_wav = parsed_args.input
    fourier = OwnFourier()

    # Read the wav file and plot it
    fourier.read_wav(input_wav)
    fourier.plot_wav()


def _run_fourier(parsed_args):
    """
    Runs the fourier task. The fourier task is used to plot the fourier transformation of the wav file.
    :param parsed_args: The parsed arguments.
    :return: None
    """
    # Get the arguments
    input_wav = parsed_args.input
    own = parsed_args.own
    xlim = parsed_args.xlim

    # Check whether to use the own implementation or the library implementation
    if own:
        fourier = OwnFourier()
    else:
        fourier = NpFourier()

    # Read the wav file and transform it
    fourier.read_wav(input_wav)
    fourier_data, frequencies = fourier.transform()
    fourier.plot(fourier_data, frequencies, x_lim=xlim)


def _run_gabor(parsed_args: Any) -> None:
    """
    Runs the gabor task. The gabor task is used to plot the gabor transformation of the wav file using the
    library pcolormesh function.
    :param parsed_args: The parsed arguments.
    :return: None
    """
    # Get the arguments
    input_wav = parsed_args.input
    own = parsed_args.own
    num_data = parsed_args.num_data
    nfft = parsed_args.nfft
    noverlap = parsed_args.noverlap
    ylim = parsed_args.ylim

    # Check whether to use the own implementation or the library implementation
    if own:
        gabor = OwnGabor()
    else:
        gabor = NpGabor()

    # Read the wav file and transform it
    gabor.read_wav(input_wav)
    spectrum, frequencies, t = gabor.transform(num_data=num_data, nfft=nfft, noverlap=noverlap)
    gabor.plot(spectrum, frequencies, t, y_lim=ylim)


def _run_specgram(parsed_args: Any) -> None:
    """
    Runs the specgram task. The specgram task is used to plot the spectrogram of the wav file using the library
    specgram function.
    :param parsed_args: The parsed arguments.
    :return: None
    """
    # Get the arguments
    input_wav = parsed_args.input
    nfft = parsed_args.nfft
    noverlap = parsed_args.noverlap
    xlim = parsed_args.xlim
    ylim = parsed_args.ylim

    # Read the wav file and transform it
    gabor = NpGabor()
    gabor.read_wav(input_wav)
    gabor.specgram(nfft=nfft, noverlap=noverlap, x_lim=xlim, y_lim=ylim)


def main(args: List[str]) -> int:
    """
    Main method of the program. The main method is used to parse the command line arguments and to execute the
    corresponding task.
    :param args: The command line arguments.
    :return: The exit code.
    """
    # Parse the command line arguments
    arg_parser = _set_up_arg_parser()
    parsed_args = arg_parser.parse_args(args)
    task = Tasks(parsed_args.command)

    # Execute the corresponding task
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
