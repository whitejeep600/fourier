import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
import scipy
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from scipy.io import wavfile
from tqdm import tqdm

N_PROCESSES = 6

# Todo adjust this to obtain an animation
#  that visually corresponds to the sound
AVERAGING_WINDOW_LEN = 32

TARGET_FPS = 90

# This is only for debugging, because calculating all the stuff
# takes a lot of time. If this is not None, it determines how many
# # video frames will be rendered.
FRAME_CUTOFF: int | None = 1024

IMG_SIZE = 512


# Fourier transform is sensitive to small changes in the audio,
# but we want to obtain a smooth visualization that looks like
# it changes continuously frame-to-frame. So we take a moving
# average of data that will go into creating each frame. We only
# average with the preceding frames, so that there is a visual
# effect of delay - sudden changes in the sound create sudden
# and instant changes in the video, are not preceded by them,
# and afterwards their influence gradually goes down to 0.
def moving_average_backwards(a: np.ndarray, length: int) -> np.ndarray:
    cumsums = np.cumsum(a, dtype=float, axis=0)
    cumsums[length:, :] -= cumsums[:-length, :]
    cumsums[length:, :] /= length
    cumsums[:length, :] /= np.arange(1, length+1)[:, None]
    return cumsums


def get_int_distances_from_array_center(
        array_size: int
) -> np.ndarray:
    center = array_size // 2
    one_dim_distance = np.abs(np.arange(array_size) - center)
    total_distance = (
            (one_dim_distance[None, :] ** 2 + one_dim_distance[:, None] ** 2) ** (1 / 2)
    ).astype(int)
    return total_distance


DISTANCES_FROM_IMG_CENTER = get_int_distances_from_array_center(IMG_SIZE)

DISTANCE_MASKS = [
    DISTANCES_FROM_IMG_CENTER == radius
    for radius in range(IMG_SIZE // 2 + 1)
]


# Writes a constant value in an array, approximately on a circle
# with the same center as the array, and a given radius.
def write_circle(
        a: np.ndarray,
        value: float,
        radius: int
) -> None:
    a[DISTANCE_MASKS[radius]] = value


# Return an uint8 array representing BGR data of the visualization.
# Of course defining this is up to our creativity.
# todo add some other processing.
#  Whatever makes it look cool, yo. We can experiment.
#  most of all, add colors
def visualize_amplitudes(
        amplitudes: np.ndarray,
        global_max_amplitude_sum: float
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
    # On each of those circles, we assign a greyscale value
    # proportional to the given amplitude.
    # This is kinda intuitive because on PSDs, higher
    # frequencies are in fact further away from the center,
    # and higher grayscale values correspond to higher
    # wave amplitudes.
    power_image = np.zeros((IMG_SIZE, IMG_SIZE))

    # todo might want to modify this for the best, most dynamic
    #  visual effect
    local_amplitude_sum = amplitudes.sum()
    brightness_scaling = (local_amplitude_sum / global_max_amplitude_sum) ** (1/2)

    # Processing that empirically works okay, not really
    # motivated by any kind of maths
    amplitudes = np.log(amplitudes)
    amplitudes -= amplitudes.min()
    amplitudes = amplitudes * 255

    # These dimensions are sensible; remove this assertion
    # if support for different ones is added
    assert (len(amplitudes) - 1) * 2 == IMG_SIZE

    for j in range(len(amplitudes)):
        write_circle(power_image, amplitudes[j], j)
    transformed = abs(np.fft.ifft2(power_image))

    # Again, empirically works to produce cool images
    transformed -= transformed.min()
    transformed = np.log(transformed)

    transformed *= 255 / transformed.max()

    # This transformation is not the most elegant ever,
    # mathematically. Buut it introduces some element
    # of chaos that works nicely on the resulting patterns,
    # making them more volatile and likely to create sharp
    # changes in brightness.
    transformed -= (transformed // 256) * 256
    transformed *= brightness_scaling

    transformed = transformed[:, :, None]
    transformed = np.repeat(transformed[:, :, None], 3, axis=2)

    transformed = transformed.astype(np.uint8)

    return transformed


def save_amplitude_data_visualization(
        amplitudes: np.ndarray,
        target_path: Path,
        n_frames_to_render: int,
        original_audio_path: Path
) -> None:

    # calculate how many frames should be taken based on the
    # audio length and the target fps. Here, take a subset
    # (linspace) of all the amplitudes
    amplitudes_selection = np.linspace(0, len(amplitudes)-1, n_frames_to_render).astype(int)
    amplitudes = amplitudes[amplitudes_selection, :]

    amplitudes = moving_average_backwards(amplitudes, AVERAGING_WINDOW_LEN)

    if FRAME_CUTOFF is not None:
        amplitudes = amplitudes[:FRAME_CUTOFF]

    max_sum_amplitude = max([frame_amplitudes.sum() for frame_amplitudes in amplitudes])

    visualize_amplitudes_partial = partial(
        visualize_amplitudes,
        global_max_amplitude_sum=max_sum_amplitude
    )

    pool = multiprocessing.Pool(N_PROCESSES)

    amplitude_visualizations = []
    for amplitude_visualization in tqdm(
            pool.imap(visualize_amplitudes_partial, amplitudes),
            desc="Creating video frames...",
            total=len(amplitudes)
    ):
        amplitude_visualizations.append(amplitude_visualization)

    video_clip = ImageSequenceClip(amplitude_visualizations, TARGET_FPS)

    audio_clip = AudioFileClip(str(original_audio_path))
    if FRAME_CUTOFF is not None:
        target_len_seconds = FRAME_CUTOFF / TARGET_FPS
        audio_clip = audio_clip.subclip(0, target_len_seconds)

    composite_audio_clip = CompositeAudioClip([audio_clip])
    video_clip.audio = composite_audio_clip
    video_clip.write_videofile(str(target_path), fps=TARGET_FPS)


def main():
    audio_path = Path("data/audio/source.wav")
    target_mp4_path = Path("data/video/final.mp4")
    target_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    samplerate, stereo_signal = wavfile.read(audio_path)
    mono_signal = stereo_signal.mean(axis=1)
    audio_len_seconds = len(mono_signal) / samplerate

    # If the original audio has n samples, this returns
    # an array of shape [IMG_SIZE//2, n // (IMG_SIZE//2) + 2] (I think), with
    # the ith row representing the fourier transform data of the
    # ith segment, and each segment being a sequence of IMG_SIZE // 2 samples
    # in the original audio.
    _, _, stft = scipy.signal.stft(mono_signal, nperseg=IMG_SIZE)

    # We want to process fragment by fragment so this indexing
    # is more intuitive
    stft = stft.T

    # We ignore the phase information and only take the amplitude
    # information of every frequency.
    all_amplitudes = np.abs(stft)

    n_frames_to_render = int(TARGET_FPS * audio_len_seconds)

    save_amplitude_data_visualization(
        all_amplitudes, target_mp4_path, n_frames_to_render, audio_path
    )


if __name__ == '__main__':
    main()
