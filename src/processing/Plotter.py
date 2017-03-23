import itertools
import matplotlib.pyplot as plt
import numpy as np

from constants import constants


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
            subplot = figure.add_subplot(5, 3, counter)
            subplot.xlabel("Hz")
            subplot.ylabel("uV")
            subplot.set_title(sensor)
            values = sensors_data[sensor]
            subplot.bar(range(constants.EEG_MIN_FREQ, len(values)+1), values, alpha=0.5)

        plt.suptitle(self.plot_name)
        plt.tight_layout()
        plt.show()

    def plot_condorcet(self, n):
        x = []
        y = []
        for N in range(1, n, 2):
            if N % 2 == 1:
                k = np.ceil(N / 2.0)
            else:
                k = N / 2.0 + 1
            p = 0.75
            sum = 0
            for i in range(int(k), N + 1):
                sum += len(list(itertools.combinations(range(N), i))) * (p ** i) * (1 - p) ** (N - i)
            x.append(N)
            y.append(sum)
            print str(N) + ":" + str(sum)
        plt.plot(x,y)
        plt.xlabel("Number of voters")
        plt.ylabel("Probability")
        plt.ylim(ymin=0.75)
        plt.grid()
        plt.show()


