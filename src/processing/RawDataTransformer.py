from math import floor


import numpy as np
from wyrm.processing import stft

from constants import constants as c

class RawDataTransformer:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def transform(self):
        sensors_data = self.get_sensor_results()
        result = {}
        for sensor in sensors_data:
            signal_data = sensors_data[sensor]
            freqs, amplitudes = self.window_fft_transform(signal_data)
            signal_freq_domains = []
            for i, val in enumerate(amplitudes):
                signal_freq_domain = self.get_freq_domain(val, freqs)
                signal_freq_domains.append(signal_freq_domain)

            result[sensor] = signal_freq_domains

        return result


    def get_freq_domain(self, amplitudes, freqs):
        freq_values = []
        for i in range(c.EEG_MIN_FREQ, c.EEG_MAX_FREQ):
            amplitude =np.interp(i, freqs, amplitudes)
            rounded_amplitude = round(amplitude, c.AMPLITUDE_VALUE_DIGITS_AFTER_ZERO)
            freq_values.append(rounded_amplitude)
        return freq_values


    def get_sensor_results(self):
        result_by_sensors = {}
        for sensor_name in c.SENSORS:
            samples = []
            for r in self.raw_data:
                sample = getattr(r, sensor_name)[0]
                samples.append(sample)
            result_by_sensors[sensor_name] = samples

        return result_by_sensors


    def window_fft_transform(self, signal_data):
        window_size_s = 1
        window_size = int(window_size_s/1*128.0)
        freqs = np.linspace(0, 128/2, window_size)
        all_windows_results = stft(signal_data, window_size)
        amplitudes = []
        for i, val in enumerate(all_windows_results):
            amplitudes.append(abs(val))

        return freqs, amplitudes

