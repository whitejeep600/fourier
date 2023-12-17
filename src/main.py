from pathlib import Path

import cv2
import numpy as np
import scipy

from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.io import wavfile
from tqdm import tqdm


AVERAGING_WINDOW_LEN = 256


def moving_average(x, length):
    ret = np.cumsum(x, dtype=float, axis=0)
    ret[length:, :] = ret[length:, :] - ret[:-length, :]
    return ret[length - 1:, :] / length


def write_circle(array, value, radius):
    h, w = array.shape
    assert h == w
    size = h
    center = size // 2
    x_distance = np.abs(np.arange(size) - center)
    y_distance = np.abs(np.arange(size) - center)
    total_distance = ((x_distance[None, :] ** 2 + y_distance[:, None] ** 2) ** (1/2)).astype(int)
    array[total_distance == radius] = value


def main():
    audio_path = Path("data/audio/source.wav")
    samplerate, stereo_signal = wavfile.read(audio_path)
    mono_signal = stereo_signal.mean(axis=1)
    audio_len_seconds = len(mono_signal) / samplerate
    _, _, stft = scipy.signal.stft(mono_signal)
    stft = stft.T

    all_amplitudes = np.abs(stft)
    all_amplitudes = moving_average(all_amplitudes, AVERAGING_WINDOW_LEN)

    target_video_fps = len(all_amplitudes) / audio_len_seconds
    target_video_path = "output.avi"
    video = cv2.VideoWriter(
        target_video_path,
        cv2.VideoWriter_fourcc(*'MPEG'),
        target_video_fps,
        (256, 256),
        False
    )

    frame_cutoff = int(target_video_fps * 2)
    for i in tqdm(range(len(all_amplitudes))[:frame_cutoff]):
        these_amplitudes = all_amplitudes[i]
        power_image = np.zeros((256, 256))
        these_amplitudes = np.log(these_amplitudes)
        these_amplitudes -= these_amplitudes.min()
        these_amplitudes = (these_amplitudes * 255).astype(int)

        for j in range(len(these_amplitudes)):
            write_circle(power_image, these_amplitudes[j], j)
        transformed = abs(np.fft.ifft2(power_image))
        transformed = np.log(transformed)
        transformed *= 255 / transformed.max()
        transformed = transformed.astype(int)

        video.write(transformed.astype(np.uint8))

    cv2.destroyAllWindows()
    video.release()

    video_clip = VideoFileClip(target_video_path)
    video_clip.write_videofile("final.mp4", fps=target_video_fps)
    pass


if __name__ == '__main__':
    main()
