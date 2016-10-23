from math import floor
from threading import Thread

import time

from numpy.fft import fftfreq, fft, fftshift

import constants as c
from emotiv.EmotivMeasurementThread import EmotivMeasurementThread
from wyrm.processing import *
import matplotlib.pyplot as plt

class LearningService(object):
    def start(self):
        print "Please concentrate on the following for " + str(c.USER_TRAINING_DURATION_S) + "s :"
        task = c.CLASSIFICATOR_LEFT
        print task

        #s = self.__get_sensor_results()[result_by_sensors.keys()[0]]
        s=[8210, 8290, 8290, 8296, 8251, 8336, 8260, 8339, 8229, 8316, 8265, 8302, 8276, 8281, 8307, 8280, 8317, 8343, 8253, 8336, 8254, 8286, 8282, 8258, 8276, 8301, 8240, 8292, 8247, 8215, 8260, 8213, 8221, 8250, 8186, 8214, 8221, 8214, 8261, 8245, 8243, 8180, 8235, 8261, 8179, 8274, 8201, 8229, 8181, 8260, 8229, 8199, 8244, 8239, 8233, 8271, 8269, 8255, 8276, 8266, 8258, 8250, 8247, 8232, 8257, 8254, 8240, 8219, 8226, 8200, 8197, 8143, 8165, 8201, 8191, 8193, 8222, 8230, 8240, 8258, 8274, 8277, 8281, 8290, 8278, 8303, 8296, 8287, 8290, 8295, 8280, 8306, 8316, 8317, 8316, 8327, 8324, 8319, 8310, 8344, 8334, 8330, 8336, 8325, 8332, 8342, 8348, 8332, 8358, 8345, 8345, 8368, 8355, 8353, 8399, 8383, 8376, 8361, 8353, 8355, 8349, 8365, 8366, 8376, 8356, 8350, 8354, 8359, 8344, 8350, 8364, 8342, 8350, 8348, 8346, 8354, 8341, 8345, 8379, 8382, 8390, 8419, 8445, 8436, 8435, 8419, 8408, 8380, 8381, 8370, 8381, 8373, 8361, 8355, 8351, 8366, 8374, 8375, 8381, 8376, 8380, 8356, 8351, 8378, 8346, 8351, 8357, 8364, 8349, 8347, 8350, 8345, 8356, 8346, 8344, 8350, 8356, 8375, 8376, 8356, 8348, 8351, 8364, 8373, 8353, 8347, 8367, 8377, 8374, 8366, 8342, 8370, 8348, 8354, 8325, 8356, 8331, 8343, 8302, 8397, 8482, 8478, 8426, 8442, 8408, 8443, 8378, 8422, 8331, 8428, 8324, 8382, 8363, 8366, 8369, 8352, 8322, 8353, 8335, 8351, 8330, 8360, 8327, 8357, 8300, 8380, 8322, 8354, 8313, 8333, 8315, 8322, 8312, 8278, 8312, 8296, 8318, 8286, 8310, 8311, 8322, 8293, 8251, 8296, 8234, 8345, 8308, 8326, 8294, 8407, 8319, 8330, 8327, 8297, 8367, 8303, 8325, 8290, 8349, 8262, 8313, 8312, 8328, 8296, 8278, 8266, 8333, 8267, 8334, 8306, 8282, 8264, 8277, 8275, 8309, 8325, 8284, 8318, 8324, 8317, 8280, 8355, 8303, 8320, 8345, 8316, 8318, 8329, 8347, 8315, 8359, 8338, 8343, 8318, 8334, 8323, 8334, 8310, 8288, 8310, 8317, 8319, 8307, 8301, 8301, 8278, 8291, 8278, 8323, 8339, 8330, 8319, 8328, 8348, 8422, 8374, 8397, 8398, 8282]
        print ("Sample length " + str(len(s)))

        yf = fft(s)
        size = len(s)
        print(size)
        freqs = fftfreq(size)
        frate =128.0
        freq_in_hertz = [abs(f * frate) for f in freqs]
        print(freq_in_hertz)

        #xf = fftshift(xf)
        #yplot = fftshift(yf)
        plt.plot(freqs, 1.0 / size * np.abs(yf))
        plt.grid()
        plt.show()

        print len(s)


    def __get_sensor_results(self):
        thread = EmotivMeasurementThread()
        thread.start()
        time.sleep(c.USER_TRAINING_DURATION_S)
        result = thread.stop()
        result_by_sensors = {}
        for sensor_name in c.SENSORS:
            samples = [r[sensor_name] for r in result]
            result_by_sensors[sensor_name] = samples
        return result_by_sensors

    def __ft_window_samples(self, result):
        ft_window_width_time = 0.2
        ft_windows_per_training = c.USER_TRAINING_DURATION_S / ft_window_width_time
        ft_window_width_samples = int(floor(len(result) / ft_windows_per_training))
        return ft_window_width_samples

