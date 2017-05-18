import time

import gevent
from constants import constants as c

class Sampler:

    def __init__(self, queue):
        self.queue = queue

    def get_samples(self, n_seconds):
        # time.sleep(c.USER_TRAINING_DURATION_S)
        while not self.queue.empty():
            self.queue.get()
        start = int(round(time.time() * 1000))
        n_samples = c.HEADSET_FREQ * n_seconds

        counter = 0
        result = []
        while counter < n_samples:
            #time.sleep(0.005)
            if not self.queue.empty():
                result.append(self.queue.get())
                counter += 1
            gevent.sleep(0.001)

        # time.sleep(10)
        stop = int(round(time.time() * 1000))

        print "Time spent " + str(stop - start)

        return result


