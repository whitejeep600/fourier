# fourierdoscope

This is a small audiovisual project devoted to automatic creation of interesting visual representations of music. Its basic idea is to apply the Fourier transform, then create a 2D image based on the obtained data, then treat that as a power spectral density image and apply an inverse Fourier transform to get the final visualization. This results in mathematically and aesthetically pleasing, kaleidoscopic patterns.

Given the path to a .wav file with a piece of music, main.py will create an .mp4 file with the music and accompanying visualization.
