#! /usr/bin/env python3
"""Beat detection using librosa."""
import argparse
import librosa

hop_length = 1024


def beat_detection(filename):
    """Beat detection using librosa."""
    # y  = loaded audio as waveform
    # sr = sampling rate
    y, sr = librosa.load(filename, sr=44100)

    # Beat track using the entire signal. Discard the beat frames.
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    print(f"Estimated tempo: {tempo:.2f} beats per minute.")

    # Separate harmonics and percussives into two waveforms.
    _, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal. Discard the beat frames.
    tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
    print(f"Estimated tempo (using percussive signal): {tempo:.2f} beats per minute.")


def main():
    """Parameter handling main method."""
    parser = argparse.ArgumentParser(description="Beat detector using librosa.")
    parser.add_argument("-f", "--filename", type=str, help="mp3 or wav file to analyze")
    args = vars(parser.parse_args())
    filename = args["filename"]
    if filename is not None:
        print(f"Filename: {filename}")
        beat_detection(filename)


if __name__ == "__main__":
    main()
