from multiprocessing import Process, Queue

from sklearn.datasets import load_iris

from LearningService import LearningService
from src.TestingService import TestingService
from src.TrainingService import TrainingService
from src.analyzing.RFTLearner import RFTLearner
from src.emotiv.Reader import Reader
from src.emotiv.Sampler import Sampler
from src.processing.RawDataTransformer import RawDataTransformer
from src.util.DataStorer import DataStorer

import pandas as pd
import numpy as np

from src.util.TimeStampGenerator import get_timestamp




def measure():
    print "This is BCI application for prediction user thoughts"
    packet_queue = Queue()
    Process(target=emotiv_start_reader, args=(packet_queue,)).start()
    raw_data = Sampler(packet_queue).get_samples(20)
    freq_domains = RawDataTransformer(raw_data).transform()
    windows_number = len(freq_domains.values()[0])
    ts = get_timestamp()
    samples = []
    for w in range(windows_number):
        sample = ['Right']
        for sensor in freq_domains:
            sample += freq_domains[sensor][w]
        samples.append(sample)

    DataStorer.store(samples, ts)


if __name__ == "__main__":
    #TestingService().start()
    TrainingService().start()
    exit()
    while(True):
        LearningService().start()
    exit()
    answer = raw_input("If you want to continue with learning mode please type '1', with training mode '2'")
    if answer == "1":
        print "Switched to learning mode"
        LearningService().start()
    if answer == "2":
        print "Switched to training mode"
        TrainingService().start()
    else:
        print "Switched to test mode"


