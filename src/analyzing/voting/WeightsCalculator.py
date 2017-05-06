from math import floor

import numpy as np

from TrainingService import TrainingService
from analyzing.AccuracyCalculator import AccuracyCalculator
from analyzing.DataDivider import DataDivider
from analyzing.voting.MajorityProbabilitiesToClassesConverter import MajorityProbabilitiesToClassesConverter
from analyzing.voting.WeightsVotingHandler import WeightsVotingHandler
from constants import constants
import matplotlib.pyplot as plt

class WeightsCalculator(object):

    def calculate_weights(self, probabilities):
        prediction_result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
        weights = self.find_best_weights_for_target(prediction_result)
        return weights

    def calculate_weights1(self, training_data):
        raise Exception("Deprecated")
        chunks = 3
        training_chunks = 2
        data_size = len(training_data)
        chunk_size = int(floor(data_size / chunks))
        extra_elements = data_size % chunk_size
        dataset_end = data_size - extra_elements
        chunk_borders = range(0, data_size, chunk_size)
        chunk_borders.append(dataset_end)

        weights_from_chunks = []

        for i in range(chunks):
            current_chunk = np.concatenate([training_data[:chunk_borders[i]], training_data[chunk_borders[i + 1]:],
                                            training_data[chunk_borders[i]:chunk_borders[i + 1]]])

            train_data, test_data = DataDivider().split_data(current_chunk, 0, training_chunks * chunk_size)
            probabilities, clf = TrainingService().train_and_test(train_data, test_data)
            prediction_result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
            weights = self.find_best_weights_for_target(prediction_result)
            weights_from_chunks.append(weights)

        #print "Weights from chunks" + str(weights_from_chunks)
        avg_weights = {}

        for i, val in enumerate(weights_from_chunks):
            weight = weights_from_chunks[i]
            for target in weight:
                avg_weights[target] = (avg_weights.get(target, weight[target]) + weight[target]) / 2.0

        return avg_weights

    def find_best_weights_for_target(self, prediction_result):
        number_of_samples = constants.SAMPLES_PER_TEST_SESSION
        weights_to_test = [(x, number_of_samples - x) for x in range(number_of_samples+1)]
        #weights_to_test = [(x * 0.01, 200 * 0.01 - x * 0.01) for x in range(30, 170, 1)]
        max_accuracy = 0
        best_weights = None
        accs=[]
        t=[]
        for w in weights_to_test:
            weights = {prediction_result.keys()[0]: w[0], prediction_result.keys()[1]: w[1]}
            acc = self.find_accuracy_for_weights(prediction_result, weights)
            accs.append(acc)
            t.append(weights.values()[0])
            if acc > max_accuracy:
                max_accuracy = acc
                best_weights = weights

        #self.plot(t, accs)
        return best_weights

    def find_accuracy_for_weights(self, prediction_result, weights):
        new_result = WeightsVotingHandler().vote_with_weights(prediction_result, weights)
        return AccuracyCalculator().get_accuracy(new_result)


    def plot(self, thresholds, accuracies):

        plt.bar(thresholds, accuracies)
        plt.xticks([x + .5 for x in thresholds], thresholds)
        plt.xlabel('Threshold', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        #plt.xticks(range(0,18,1))

        plt.show()
