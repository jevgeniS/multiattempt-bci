from multiprocessing import Process, Queue

import constants as c
from src.emotiv.Reader import Reader
from src.emotiv.Sampler import Sampler
from src.processing.RawDataTransformer import RawDataTransformer
from src.util.DataStorer import DataStorer
from src.util.TimeStampGenerator import get_timestamp


def emotiv_start_reader(queue):
    Reader.read(queue)

class LearningService(object):
    def select_target(self):
        print "Select target typing the number of a target"
        for key in c.TARGETS.keys():
            print key + ":" + c.TARGETS[key]
        answer = raw_input()

        return c.TARGETS[answer]


    def start(self):
        target = self.select_target()
        print "Please concentrate on the '"+target+"' for " + str(c.USER_TRAINING_DURATION_S) + "s :"

        packet_queue = Queue()
        process = Process(target=emotiv_start_reader, args=(packet_queue,))
        process.start()
        raw_data = Sampler(packet_queue).get_samples(10)
        process.terminate()
        freq_domains = RawDataTransformer(raw_data).transform()
        windows_number = len(freq_domains.values()[0])
        ts = get_timestamp()
        samples = []
        for w in range(windows_number):
            sample = [target]
            for sensor in freq_domains:
                sample += freq_domains[sensor][w]
            samples.append(sample)

        DataStorer.store(samples, ts)






