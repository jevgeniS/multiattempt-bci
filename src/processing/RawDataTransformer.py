from math import floor

import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from wyrm.processing import stft


class RawDataTransformer:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def plain_fft_transform(self, visualize=False):
        print
        print self.raw_data
        yf = fft(self.raw_data)
        n = len(self.raw_data)
        print "Length " + str(n)
        print (min(self.raw_data))
        print (max(self.raw_data))
        n_frequencies_below_niquist = int(floor(n / 2.0))
        freqs = fftfreq(n, 1.0/128.0)[:n_frequencies_below_niquist]


        if visualize:
            fig, ax = plt.subplots()
            ax.plot(freqs[1:], abs(yf[:n_frequencies_below_niquist])[1:])

            plt.grid()
            plt.show()

    def window_fft_transform(self, visualize=False):
        #Does not work properly
        yf = stft(self.raw_data, 200)[0]
        n=len(yf)
        n_frequencies_below_niquist = int(floor(n / 2.0))
        freqs = fftfreq(n)[:n_frequencies_below_niquist]

        if visualize:
            fig, ax = plt.subplots()
            ax.plot(freqs, abs(yf[:n_frequencies_below_niquist]))
            plt.grid()
            plt.show()