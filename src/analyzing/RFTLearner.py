from random import shuffle

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src import constants


class RFTLearner:

    def __init__(self):
        self.classifier = None
        self.accuracy = 0.0


    def predict(self, sample):
        return self.classifier.predict(sample)





    def setup_classifier(self, training_data):
        #training_data, test_data = self.split_data(data)
        self.classifier = self.train(training_data)


    def setup_classifier_with_cross_validation(self, data):
        shuffle(data)
        parts_n = 5
        part_elements_n = int(len(data)/parts_n)
        for i in range(parts_n):
            test_data_chunk_start= i*part_elements_n
            test_data_chunk_end = i*part_elements_n+part_elements_n
            test_data = np.array(data[test_data_chunk_start:test_data_chunk_end])
            training_data_first_part = data[:test_data_chunk_start]
            training_data_second_part = data[test_data_chunk_end:]
            training_data = np.array(training_data_first_part+training_data_second_part)

            clf, accuracy = self.train(training_data, test_data)

            if self.accuracy < accuracy:
                self.accuracy = accuracy
                self.classifier = clf

        print "Trained classifier with accuracy " + str(self.accuracy) + "%"
        return self.accuracy

    def train(self, training_data):
        training_targets = training_data[:, 0]
        training_features = training_data[:, 1:]

        clf = RandomForestClassifier(n_jobs=2)
        clf.fit(training_features, training_targets)

        return clf

    def get_accuracy(self, test_data):
        test_targets = test_data[:, 0]
        test_features = test_data[:, 1:]
        predicted_targets = self.classifier.predict(test_features)
        accuracy = self.classification_accuracy(test_targets, predicted_targets)
        return accuracy

    def classification_accuracy(self, actual_targets, predicted_targets):
        n = float(len(actual_targets))
        matches = 0.0
        for i, target in enumerate(actual_targets):
            if target == predicted_targets[i]:
                matches += 1.0

        return matches/n*100



