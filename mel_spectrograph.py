#! /usr/bin/env python3
"""Mel spectrograph using librosa."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import librosa  # noqa: E402
import librosa.display  # noqa: E402

hop_length = 1024


def mel_spectrograph(filename):
    """Mel spectrograph using librosa."""
    # y  = loaded audio as waveform
    # sr = sampling rate
    y, sr = librosa.load(filename, sr=44100)

    # Mel-scaled power (energy-squared) spectrogram.
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). Use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(12, 4))

    # Display the spectrogram on a mel scale.
    # sample rate and hop length parameters are used to render the time axis.
    librosa.display.specshow(log_S, sr=sr, x_axis="time", y_axis="mel")

    plt.title("mel power spectrogram")
    plt.colorbar(format="%+02.0f dB")
    plt.tight_layout()
    plt.show()


def mel_spectrograph_by_source(filename):
    """Mel spectrograph by source using librosa."""
    # y  = loaded audio as waveform
    # sr = sampling rate
    y, sr = librosa.load(filename, sr=44100)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Mel-scaled power (energy-squared) spectrogram.
    S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr)
    S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

    # Convert to log scale (dB). Use the peak power (max) as reference.
    log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
    log_Sp = librosa.power_to_db(S_percussive, ref=np.max)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(log_Sh, sr=sr, x_axis="time", y_axis="mel")
    plt.title("mel power spectrogram (Harmonic)")
    plt.colorbar(format="%+02.0f dB")
    plt.subplot(2, 1, 2)
    librosa.display.specshow(log_Sp, sr=sr, x_axis="time", y_axis="mel")
    plt.title("mel power spectrogram (Percussive)")
    plt.colorbar(format="%+02.0f dB")
    plt.tight_layout()
    plt.show()


def main():
    """Parameter handling main method."""
    parser = argparse.ArgumentParser(description="Mel spectrograph using librosa.")
    parser.add_argument("-f", "--filename", type=str, help="mp3 or wav file to analyze")
    args = vars(parser.parse_args())
    filename = args["filename"]
    if filename is not None:
        print(f"Filename: {filename}")
        mel_spectrograph(filename)
        mel_spectrograph_by_source(filename)


if __name__ == "__main__":
    main()
