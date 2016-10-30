import time
from math import floor

from emokit.emotiv import Emotiv

import constants as c
from processing.RawDataTransformer import RawDataTransformer


class LearningService(object):
    def start(self):
        print "Please concentrate on the following for " + str(c.USER_TRAINING_DURATION_S) + "s :"
        task = c.CLASSIFICATOR_LEFT
        print task

        s = self.__get_sensor_results
        RawDataTransformer(s["AF3"]).plain_fft_transform(True)


        print ("Sample length " + str(len(s)))





    @property
    def __get_sensor_results(self):
        result = []
        with Emotiv(display_output=True, verbose=True, is_research=False, write_values=False,
                    write_encrypted=True) as headset:

            time.sleep(c.USER_TRAINING_DURATION_S)
            headset.stop()
            while True:
                packet = headset.dequeue()
                if packet is None:
                    break
                if packet is not None:
                    result.append(packet)

            result_by_sensors = {}
            for sensor_name in c.SENSORS:
                samples = []
                for r in result:
                    sample = getattr(r, sensor_name)[0]
                    samples.append(sample)
                result_by_sensors[sensor_name] = samples

            return result_by_sensors

    def __ft_window_samples(self, result):
        ft_window_width_time = 0.2
        ft_windows_per_training = c.USER_TRAINING_DURATION_S / ft_window_width_time
        ft_window_width_samples = int(floor(len(result) / ft_windows_per_training))
        return ft_window_width_samples

