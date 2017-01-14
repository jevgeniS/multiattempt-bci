from random import shuffle
import numpy as np
from src import constants
from src.analyzing.RFTLearner import RFTLearner
from src.util.DataStorer import DataStorer


class TrainingService(object):
    def start(self):
        data = DataStorer.read()
        n = 7
        count = 0
        learner = RFTLearner()

        step = 0

        for i in range(n):
            train_data, test_data = self.split_data_percent(data, step)
            #train_data, test_data = self.split_data(data)
            step += 10
            learner.setup_classifier(train_data)
            accuracy = learner.get_accuracy(test_data)
            print accuracy
            count += accuracy

        print "Average:"+str(count / n)


    def split_data(self, data):
        #shuffle(data)
        training_border_index = int(constants.TRAINING_DATA_PERCENTAGE / 100.0 * len(data))

        training_data = np.array(data[:training_border_index])
        test_data = np.array(data[training_border_index:])

        return training_data, test_data

    def split_data_percent(self, data, test_data_start_percent):

        test_data_start = int(test_data_start_percent/100.0*len(data))
        test_data_percent = 100 - int(constants.TRAINING_DATA_PERCENTAGE)
        test_data_len = int(test_data_percent/100.0*len(data))
        test_data_end = test_data_start+test_data_len
        test_data = np.array(data[test_data_start:test_data_end])
        training_data = np.array(data[:test_data_start] + data[test_data_end:])

        return training_data, test_data