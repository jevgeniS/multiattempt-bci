import matplotlib.pyplot as plt
import numpy as np

from src import constants



class Plotter:

    def __init__(self, plot_name=None):
        self.plot_name = plot_name

    def plot_avg(self, amplitudes):
        amplitudes = np.array(amplitudes).transpose()
        amplitudes_without_targets = amplitudes[1:].astype(np.float)
        sensors_data={}
        index = 0
        for i, sensor in enumerate(constants.SENSORS):
            sensor_freqs=[]
            for freq in range(constants.EEG_MIN_FREQ, constants.EEG_MAX_FREQ):
                sensor_freqs.append(np.average(amplitudes_without_targets[index]))
                index += 1
            sensors_data[sensor] = sensor_freqs
        self.plot(sensors_data)


    def plot(self, sensors_data):
        figure = plt.figure()
        counter = 0
        for sensor in sensors_data:
            counter += 1
            subplot = figure.add_subplot(4, 4, counter)
            subplot.set_title(sensor)
            values = sensors_data[sensor]
            subplot.bar(range(constants.EEG_MIN_FREQ, len(values)+1), values, alpha=0.5)
        plt.suptitle(self.plot_name)
        plt.show()

