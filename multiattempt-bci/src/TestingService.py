from itertools import groupby
from operator import itemgetter

import numpy as np
import constants as c
from BaseMeasuringService import BaseMeasuringService
from analyzing.CrossValidator import CrossValidator
from trainers.VoteTrainingService import VoteTrainingService
from util.DataStorer import DataStorer
from util.TimeStampGenerator import get_timestamp


class TestingService(BaseMeasuringService):

    def __init__(self, train_data_storer):
        super(TestingService, self).__init__()
        self.test_data_storer=self.create_testdata_storer()
        self.vote_training_service = VoteTrainingService(train_data_storer, CrossValidator())
        self.vote_training_service.train()

    def get_number_of_samples_per_session(self):
        return c.constants.USER_TEST_SAMPLES

    def process_prediction_data(self, samples):
        answer = raw_input("Type X do skip save of current try")

        if answer != 'X':
            self.test_data_storer.store(samples, get_timestamp())
        self.vote_training_service.test(np.array(samples))

    def calculate_details(self):
        test_data = self.test_data_storer.read_with_timestamps()
        test_data_with_labels = self.reduce_sample_length_to_required(test_data)
        print "Test data length after filtering by juror numbers: "+str(len(test_data_with_labels))
        labels= test_data_with_labels[:,0]
        for t in c.constants.TARGETS.values():
            print "Samples count for "+t+":"+str(len([x for x in labels if x==t]))
        print
        self.vote_training_service.test(test_data_with_labels)

    def reduce_sample_length_to_required(self, samples):
        groups=groupby(samples, key=itemgetter(0))
        result=[]
        chunk_size = c.constants.USER_TEST_SAMPLES * c.constants.SAMPLES_PER_TEST_SESSION
        for i, g in groups:
            vals=list(g)
            if len(vals)>=chunk_size:
                result=result+vals[:chunk_size]

        return np.array(result)[:, 1:]

    def create_testdata_storer(self):
        storer = DataStorer(c.constants.TEST_DATA_FILE)
        return storer

