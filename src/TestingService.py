from multiprocessing import Process, Queue

from analyzing.RFTLearner import RFTLearner
from emotiv.Reader import Reader
from emotiv.Sampler import Sampler
from processing.RawDataTransformer import RawDataTransformer
from util.DataStorer import DataStorer


def emotiv_start_reader(queue):
    Reader.read(queue)

class TestingService(object):
    def start(self):
        packet_queue = Queue()
        clf = RFTLearner()
        clf.setup_classifier(DataStorer().read())
        process = Process(target=emotiv_start_reader, args=(packet_queue,))
        process.start()
        while (True):
            raw_data = Sampler(packet_queue).get_samples(2)
            freq_domains = RawDataTransformer(raw_data).transform()
            windows_number = len(freq_domains.values()[0])
            samples = []
            for w in range(windows_number):
                sample = []
                for sensor in freq_domains:
                    sample += freq_domains[sensor][w]
                samples.append(sample)
            result = clf.predict(samples)
            print result

        process.terminate()
