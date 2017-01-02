from random import shuffle

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src import constants


class RFTLearner:

    def predict(self, sample):
        return self.classifier.predict(sample)


    def setup_classifier(self, data):
        shuffle(data)
        training_border_index = int(constants.TRAINING_DATA_PERCENTAGE/100.0*len(data))

        training_data = np.array(data[:training_border_index])
        test_data = np.array(data[training_border_index:])

        training_targets = training_data[:, 0]
        training_features = training_data[:, 1:]
        test_targets = test_data[:, 0]
        test_features = test_data[:, 1:]
        clf = RandomForestClassifier(n_jobs=2)
        clf.fit(training_features, training_targets)

        predicted_targets = clf.predict(test_features)
        accuracy = self.classification_accuracy(test_targets, predicted_targets)

        print "Created and trained classifier with accuracy "+str(accuracy)+"%"
        self.classifier = clf

        return accuracy


    def classification_accuracy(self, actual_targets, predicted_targets):
        n = float(len(actual_targets))
        matches = 0.0
        for i, target in enumerate(actual_targets):
            if target == predicted_targets[i]:
                matches += 1.0

        return matches/n*100



