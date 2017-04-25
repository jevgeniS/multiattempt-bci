from random import shuffle

import numpy as np

from constants import constants
from analyzing.RfLearner import RfLearner
from sklearn.metrics import confusion_matrix

class TrainingService(object):

    def train(self):
        raise NotImplementedError("Please Implement this method")

    def train_random_train_data(self, data):
        print "Random train data train"
        n = 10
        count = 0
        learner = RfLearner()

        for i in range(n):
            d = data[:]
            shuffle(d)
            training_border_index = int(constants.TRAINING_DATA_PERCENTAGE / 100.0 * len(data))
            train_data, test_data = self.split_data(d, 0, training_border_index)
            learner.setup_classifier(train_data)
            accuracy = learner.get_accuracy(test_data)
            #print accuracy
            count += accuracy

        print "Average:" + str(count / n)

    def train_first_chunk_as_train_second_as_test(self, data):
        print "First chunk as train"
        learner = RfLearner()
        training_border_index = int(constants.TRAINING_DATA_PERCENTAGE / 100.0 * len(data))
        train_data, test_data = self.split_data(data, 0, training_border_index)
        test_data = self.balance_data(test_data)
        n = 1
        count = 0
        for i in range(n):
            learner.setup_classifier(train_data)
            actual_data = test_data[:, 0]
            predicted_data = learner.predict_samples(test_data)
            print(confusion_matrix(actual_data, predicted_data, ["Excitement", "Relax"]))
            accuracy = learner.classification_accuracy(actual_data, predicted_data)
            #print accuracy
            count += accuracy

        print "Average:" + str(count / n)

        n = 1
        count = 0
        for i in range(n):
            learner.setup_classifier(train_data)
            actual_data = train_data[:, 0]
            predicted_data = learner.predict_samples(train_data)
            print(confusion_matrix(actual_data, predicted_data, ["Excitement", "Relax"]))
            accuracy = learner.classification_accuracy(actual_data, predicted_data)
            #print accuracy
            count += accuracy

        print "Average:" + str(count / n)



    def train_with_sliding_window(self, data):
        print "Sliding window train"
        n = 7
        count = 0
        learner = RfLearner()

        step = 0

        for i in range(n):
            train_data, test_data = self.split_data_percent(data, step)
            # train_data, test_data = self.split_data(data)
            step += 10
            learner.setup_classifier(train_data)
            accuracy = learner.get_accuracy(test_data)
            #print accuracy
            count += accuracy

        print "Average:" + str(count / n)

    def split_data(self, data, training_start_index, training_stop_index):
        data=np.array(data)
        training_data = np.array(data[training_start_index:training_stop_index])
        test_data = np.concatenate((data[:training_start_index], data[training_stop_index:]))
        return training_data, test_data



    def split_data_percent(self, data, test_data_start_percent):

        test_data_start = int(test_data_start_percent/100.0*len(data))
        test_data_percent = 100 - int(constants.TRAINING_DATA_PERCENTAGE)
        test_data_len = int(test_data_percent/100.0*len(data))
        test_data_end = test_data_start+test_data_len
        test_data = np.array(data[test_data_start:test_data_end])
        training_data = np.array(data[:test_data_start] + data[test_data_end:])

        return training_data, test_data

    def balance_data(self, data):
        targets = data[:, 0]
        targets1 = [i for i, val in enumerate(targets) if val == constants.TARGETS.values()[0]]
        targets2 = [i for i, val in enumerate(targets) if val == constants.TARGETS.values()[1]]
        t1_len = len(targets1)
        t2_len = len(targets2)
        if t1_len < t2_len:
            return np.concatenate((data[targets1], data[targets2[:t1_len]]))
        else:
            return np.concatenate((data[targets1[:t2_len]], data[targets2]))
