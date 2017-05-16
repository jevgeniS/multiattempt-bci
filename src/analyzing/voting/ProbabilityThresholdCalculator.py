import numpy as np

from analyzing.AccuracyCalculator import AccuracyCalculator
from analyzing.voting.ProbabilitiesToClassesConverter import ProbabilitiesToClassesConverter
from constants import constants

import matplotlib.pyplot as plt

class ProbabilityThresholdCalculator(object):

    def calculate_thresholds(self, probabilities):
        max_accuracy = 0.0
        max_acc_threshold=None
        t=[]
        accs=[]
        for i in np.arange(0, 1, 0.01):
            t1 = i
            t2 = 1.0 - t1
            thresholds = {constants.TARGETS.values()[0]: t1, constants.TARGETS.values()[1]: t2}
            t.append(t1)
            results = ProbabilitiesToClassesConverter(thresholds).convert(probabilities)
            accuracy = AccuracyCalculator().get_accuracy(results)
            accs.append(accuracy)
            if accuracy>max_accuracy:
                max_accuracy=accuracy
                max_acc_threshold = thresholds

        self.plot(t,accs)

        return max_acc_threshold

    def plot(self, thresholds, accuracies):

        #plt.hist(thresholds, bins=bins, weights=accuracies)
        plt.hist(np.linspace(0, 1, 100), bins=np.linspace(0, 1, 101), weights=accuracies)
        plt.axvline(0.5, color='r', linestyle='dashed', linewidth=2)
        #plt.xticks([x + .05 for x in thresholds], thresholds)
        plt.xlabel('Threshold', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        # plt.xticks(range(0,18,1))

        plt.show()
