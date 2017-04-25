from itertools import groupby
from operator import itemgetter

import numpy as np
import constants as c
from BaseMeasuringService import BaseMeasuringService
from trainers.VoteTrainingService import VoteTrainingService
from util.DataStorer import DataStorer
from util.TimeStampGenerator import get_timestamp


class TestingService(BaseMeasuringService):

    def __init__(self, train_data_storer):
        super(TestingService, self).__init__()
        self.test_data_storer=self.create_testdata_storer()
        self.vote_training_service = VoteTrainingService(train_data_storer)
        self.vote_training_service.train()


    def get_number_of_samples_per_session(self):
        return c.constants.USER_TEST_SAMPLES

    def process_prediction_data(self, samples):
        answer = raw_input("Type X do skip save of current try")

        if answer != 'X':
            self.test_data_storer.store(samples, get_timestamp())
        self.vote_training_service.test(np.array(samples)[:, 1:])

    def calculate_target(self):
        test_data = self.test_data_storer.read_with_timestamps()
        test_data_with_labels = self.reduce_sample_length_to_required(test_data)
        labels=test_data_with_labels[:,0]
        test_data=test_data_with_labels[:,1:]
        chunks = self.split_on_testsessions(test_data)
        labels = [l[0] for l in self.split_on_testsessions(labels)]

        mv_results=[]
        wv_results=[]
        mv_hits=0
        wv_hits = 0
        for i,c in enumerate(chunks):
            res1, res2= self.vote_training_service.test(c)
            mv_results.append(res1[0])
            wv_results.append(res2[0])
            if res1[0]==labels[i]:
                mv_hits += 1
            if res2[0]==labels[i]:
                wv_hits += 1

        print "Samples amount:"+str(len(chunks))
        print "Final majority vote accuracy: "+str(mv_hits/float(len(chunks)))
        print "Final weighted vote accuracy: " + str(wv_hits / float(len(chunks)))

    def split_on_testsessions(self, data):
        chunks = []
        current_index = 0
        chunk_size = c.constants.USER_TEST_SAMPLES*c.constants.SAMPLES_PER_TEST_SESSION
        while current_index + chunk_size <= len(data):
            chunks.append(data[current_index: (current_index + chunk_size)])
            current_index += chunk_size
        return chunks

    def reduce_sample_length_to_required(self, samples):
        groups=groupby(samples, key=itemgetter(0))
        result=[]
        chunk_size = c.constants.USER_TEST_SAMPLES * c.constants.SAMPLES_PER_TEST_SESSION
        for i, g in groups:
            vals=list(g)
            result=result+vals[:chunk_size]
        return np.array(result)[:, 1:]

    def create_testdata_storer(self):
        storer = DataStorer(c.constants.TEST_DATA_FILE)
        return storer