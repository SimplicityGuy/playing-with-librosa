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


def mel_spectrograph(y, sr):
    """Compute mel spectrograph."""
    # Mel-scaled power (energy-squared) spectrogram.
    S = librosa.feature.melspectrogram(y, sr=sr)

    # Convert to log scale (dB). Use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    return S, log_S


def show_mel_spectrograph(S, sr):
    """Display mel spectrograph using librosa."""
    plt.figure(figsize=(12, 4))
    # Display the spectrogram on a mel scale.
    # Sample rate and hop length parameters are used to render the time axis.
    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel")
    plt.title("mel power spectrogram")
    plt.colorbar(format="%+02.0f dB")
    plt.tight_layout()
    plt.show()


def mel_spectrograph_by_source(y_harmonic, y_percussive, sr):
    """Compute mel spectrograph by source using librosa."""
    # Mel-scaled power (energy-squared) spectrogram.
    S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr)
    S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

    # Convert to log scale (dB). Use the peak power (max) as reference.
    log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
    log_Sp = librosa.power_to_db(S_percussive, ref=np.max)

    return S_harmonic, S_percussive, log_Sh, log_Sp


def show_mel_spectrograph_by_source(log_Sh, log_Sp, sr):
    """Display mel spectrograph by source using librosa."""
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


def chromagram(y, sr):
    """Compute chroma feature extraction."""
    # CQT-based chromagram with 36 bins-per-octave in the CQT analysis.
    # Preferably, use only the harmonic component to avoid pollution from transients.
    return librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)


def show_chromagram(C, sr):
    """Display chroma feature extraction."""
    plt.figure(figsize=(12, 4))
    # The energy in each chromatic pitch class as a function of time.
    librosa.display.specshow(C, sr=sr, x_axis="time", y_axis="chroma", vmin=0, vmax=1)
    plt.title("Chromagram")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def sound_texture(log_S, num_coeff=13):
    """Compute the mel-frequency cepstral coefficients, or sound texture."""
    # Extract the top num_coeff Mel-frequency cepstral coefficients (MFCCs).
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=num_coeff)

    # Pad on the first and second deltas.
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    return mfcc, delta_mfcc, delta2_mfcc, M


def show_sound_texture(mfcc, delta_mfcc, delta2_mfcc, sr):
    """Display the mel-frequency cepstral coefficients, or sound texture."""
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mfcc)
    plt.ylabel("MFCC")
    plt.colorbar()
    plt.subplot(3, 1, 2)
    librosa.display.specshow(delta_mfcc)
    plt.ylabel("MFCC-$\Delta$")  # noqa: W605
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(delta2_mfcc, sr=sr, x_axis="time")
    plt.ylabel("MFCC-$\Delta^2$")  # noqa: W605
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def main():
    """Parameter handling main method."""
    parser = argparse.ArgumentParser(description="Mel spectrograph using librosa.")
    parser.add_argument("-f", "--filename", type=str, required=True, help="mp3 or wav file to analyze")
    parser.add_argument("--full_mel", action="store_true", help="full mel spectrograph")
    parser.add_argument("--split_mel", action="store_true", help="per source mel spectrograph")
    parser.add_argument("--chroma", action="store_true", help="chromagram from harmonic")
    parser.add_argument("--texture", action="store_true", help="sound texture")
    args = vars(parser.parse_args())

    filename = args["filename"]
    if filename is not None:
        print(f"Filename: {filename}")
        # y  = loaded audio as waveform
        # sr = sampling rate
        y, sr = librosa.load(filename, sr=44100)

    if args["full_mel"]:
        s, log_s = mel_spectrograph(y, sr)
        show_mel_spectrograph(log_s, sr)

    if args["split_mel"]:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        _, _, log_sh, log_sp = mel_spectrograph_by_source(y_harmonic, y_percussive, sr)
        show_mel_spectrograph_by_source(log_sh, log_sp, sr)

    if args["chroma"]:
        y_harmonic, _ = librosa.effects.hpss(y)
        c = chromagram(y_harmonic, sr)
        show_chromagram(c, sr)

    if args["texture"]:
        _, log_s = mel_spectrograph(y, sr)
        mfcc, delta_mfcc, delta2_mfcc, _ = sound_texture(log_s)
        show_sound_texture(mfcc, delta_mfcc, delta2_mfcc, sr)


if __name__ == "__main__":
    main()
