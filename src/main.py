from pathlib import Path

import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.io import wavfile


def moving_average(x, length):
    ret = np.cumsum(x, dtype=float, axis=1)
    ret[length:] = ret[length:] - ret[:-length]
    return ret[length - 1:] / length


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
    frequencies, times, stft = scipy.signal.stft(mono_signal)
    stft = stft.T

    all_phases = np.angle(stft)
    all_amplitudes = np.abs(stft)

    for i in range(len(all_phases)):
        these_amplitudes = all_amplitudes[i]
        power_image = np.zeros((255, 255))
        these_amplitudes = np.log(these_amplitudes)
        these_amplitudes -= these_amplitudes.min()
        these_amplitudes = (these_amplitudes * 255).astype(int)

        for j in range(len(these_amplitudes)):
            write_circle(power_image, these_amplitudes[j], j)
        plt.imshow(power_image, cmap='gray')
        plt.show()
        transformed = abs(np.fft.ifft2(power_image))
        transformed = np.log(transformed)
        transformed *= 255 / transformed.max()
        transformed = transformed.astype(int)
        plt.imshow(transformed, cmap='gray')
        plt.show()
        if i == 5:
            break






    # moving_phases = moving_average(all_phases, 256)
    # moving_amplitudes = moving_average(all_amplitudes, 256)
    #
    # # maybe create moving averages?
    # for i in range(len(moving_phases)):
    #     phases = moving_phases[i]
    #     phases -= phases.min()
    #     amplitudes = moving_amplitudes[i]
    #     amplitudes[amplitudes == 0] = amplitudes.mean()
    #     amplitudes = np.log(amplitudes + 1)
    #     amplitudes -= amplitudes.min()
    #     phases_int = (phases / phases.max() * 127).astype(int)
    #     amplitudes_int = (amplitudes / amplitudes.max() * 127 // 2).astype(int)
    #     target = np.zeros((128, 128))
    #     target[phases_int, amplitudes_int] = 255
    #     plt.imshow(target, cmap='gray')
    #     plt.show()
    #     plt.clf()
    #     if i % 32 == 0:
    #         plt.imshow(abs(np.fft.ifft2(target)), cmap='gray')
    #
    #         plt.show()
    #         plt.clf()
    #         pass
    #
    #     # to_visualize = amplitudes[None, :] + amplitudes[:, None]
    #     # to_visualize = np.log(to_visualize)
    #     # to_visualize = to_visualize / to_visualize.max() * 256
    #     # plt.imshow(abs(np.fft.ifft2(to_visualize)), cmap='gray')
    #     # plt.show()
    #     pass

    # plt.imshow(abs(np.fft.ifft2()),
    #              cmap='gray')
    # plt.show()
    # so we get 129 frequencies, 78996 moments of time, and for each
    # moment, the value of each frequency? Well, that makes sense, for
    # stft we want the values of all the frequencies in small intervals
    # of time. Here that would be samplerate / 128 = 375, which would be
    # our fps, which is absolutely decent. Now to convert that to images
    pass


if __name__ == '__main__':
    main()
