import os
from pathlib import Path

import cv2
import numpy as np
import scipy
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.io import wavfile
from tqdm import tqdm


# Todo after adding the sound, adjust this to obtain an animation
#  that visually corresponds to the sound (hopefully an adjustment
#  here will be sufficient)
AVERAGING_WINDOW_LEN = 256

# This is only for debugging, because calculating all the stuff
# takes a lot of time. We'll only render this many video frames
FRAME_CUTOFF = 512

# Working on BGR video data
BLUE_AXIS = 0
GREEN_AXIS = 1
RED_AXIS = 2


def moving_average(a: np.ndarray, length: int) -> np.ndarray:
    cumsums = np.cumsum(a, dtype=float, axis=0)
    cumsums[length:, :] -= cumsums[:-length, :]
    return cumsums[length - 1:, :] / length


# Writes a constant value in an array, approximately on a circle
# with the same center as the array and a given radius.
def write_circle(
        a: np.ndarray,
        value: float,
        radius: int
) -> None:
    h, w = a.shape
    assert h == w, "This function is meant for square arrays. Shame on you"
    size = h
    center = size // 2
    one_dim_distance = np.abs(np.arange(size) - center)
    total_distance = (
            (one_dim_distance[None, :] ** 2 + one_dim_distance[:, None] ** 2) ** (1/2)
    ).astype(int)
    a[total_distance == radius] = value


# todo maybe decrease the lightness of the whole frame if the volume goes down
#  or make the moving average move faster for "faster" fragments

# Return an uint8 array representing BGR data of the visualization.
# Of course defining this is up to our creativity.
# todo add some other processing.
#  Whatever makes it look cool, yo. We can experiment
def visualize_amplitudes(
        amplitudes: np.ndarray
) -> np.ndarray:
    # The main idea behind this visualization is as follows.
    # We get a 1-dimensional array of frequency amplitudes,
    # sorted by increasing frequencies. We want to create
    # an image based on that (somehow). We will treat that
    # image as a power spectral density (PSD) image and
    # apply the inverse Fourier transform to it in order to
    # increase coolness (for some further explanation, vide
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm).
    # The open question is hot to create a PSD based on
    # 1-dimensional amplitude data. We have chosen to
    # map increasing frequencies to circles of increasing
    # radii on the image, centered in the center of the image.
    # On these circles, we assign a greyscale value
    # proportional to the given amplitude.
    # This is kinda intuitive because on PSDs, higher
    # frequencies are in fact further away from the center,
    # and higher grayscale values correspond to higher
    # wave amplitudes.
    img_size = (len(amplitudes) - 1) * 2
    power_image = np.zeros((img_size, img_size))

    # Processing that empirically works okay
    amplitudes = np.log(amplitudes)
    amplitudes -= amplitudes.min()
    amplitudes = amplitudes * 255

    # try to make this faster maybe, every frame takes
    # 1/6th of a second to render, that's a bit much
    for j in range(len(amplitudes)):
        write_circle(power_image, amplitudes[j], j)
    transformed = abs(np.fft.ifft2(power_image))

    # ditto
    transformed = np.log(transformed)
    transformed *= 255 / transformed.max()
    transformed = np.repeat(transformed[:, :, None], 3, axis=2)
    transformed = transformed.astype(np.uint8)
    return transformed


def save_amplitude_data_visualization_as_avi(
        amplitudes: np.ndarray,
        target_path: Path,
        target_video_fps: float
) -> None:
    img_size = ((amplitudes.shape[1]) - 1) * 2
    video = cv2.VideoWriter(
        str(target_path),
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps=target_video_fps,
        frameSize=(img_size, img_size),
        isColor=True
    )

    for frame_amplitudes in tqdm(amplitudes[:FRAME_CUTOFF]):
        amplitudes_visualization = visualize_amplitudes(frame_amplitudes)

        # Adding a single frame to the video
        video.write(amplitudes_visualization)

    cv2.destroyAllWindows()
    video.release()


def resave_with_target_fps_and_sound(
        temp_video_path: Path,
        target_video_path: Path,
        target_video_fps: float,
        audio_path: Path
) -> None:

    # again, debug
    target_len_seconds = FRAME_CUTOFF / target_video_fps

    audioclip = AudioFileClip(str(audio_path)).subclip(0, target_len_seconds)
    new_audioclip = CompositeAudioClip([audioclip])

    video_clip = VideoFileClip(str(temp_video_path))
    video_clip.audio = new_audioclip
    video_clip.write_videofile(str(target_video_path), fps=target_video_fps)
    os.system(f"rm {temp_video_path}")


def main():
    audio_path = Path("data/audio/source.wav")
    target_mp4_path = Path("data/video/final.mp4")
    target_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    # Saving video it as avi first, then loading and saving as mp4
    # with audio. I don't think CV2 makes it possible to add audio.
    temp_video_path = Path("temp.avi")

    samplerate, stereo_signal = wavfile.read(audio_path)
    mono_signal = stereo_signal.mean(axis=1)
    audio_len_seconds = len(mono_signal) / samplerate

    # If the original audio has n samples, this (by default) returns
    # an array of the shape [128, n // 128 + 2] (I think), with
    # the ith row representing the fourier transform data of the
    # ith segment, and each segment being a sequence of 128 samples
    # in the original audio.
    _, _, stft = scipy.signal.stft(mono_signal)

    # We want to process fragment by fragment so this indexing
    # is more intuitive
    stft = stft.T

    # We ignore the phase information and only take the amplitude
    # information of every frequency.
    all_amplitudes = np.abs(stft)
    target_video_fps = len(all_amplitudes) / audio_len_seconds

    # Fourier transform is sensitive to small changes in the audio,
    # but we want to obtain a smooth visualization that looks like
    # it changes continuously frame-to-frame. So we take a moving
    # average.
    all_amplitudes = moving_average(all_amplitudes, AVERAGING_WINDOW_LEN)

    save_amplitude_data_visualization_as_avi(all_amplitudes, temp_video_path, target_video_fps)
    resave_with_target_fps_and_sound(temp_video_path, target_mp4_path, target_video_fps, audio_path)


if __name__ == '__main__':
    main()
