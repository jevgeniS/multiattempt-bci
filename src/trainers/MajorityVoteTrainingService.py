from collections import Counter
from math import floor

from constants import constants
from TrainingService import TrainingService
from analyzing.RFTLearner import RFTLearner
from util.DataStorer import DataStorer
import numpy as np

class MajorityVoteTrainingService(TrainingService):

    def train(self):
        data = DataStorer.read()
        self.train_first_chunk_as_train_second_as_test_majority_vote(data)

    def train_first_chunk_as_train_second_as_test_majority_vote(self, data):
        training_border_index = int(constants.TRAINING_DATA_PERCENTAGE / 100.0 * len(data))
        train_data, test_data = self.split_data(data, 0, training_border_index)
        result = self.train_and_test(train_data, test_data)
        self.majority_vote(result.values()[0], result.keys()[0])
        self.majority_vote(result.values()[1], result.keys()[1])
        print(self.calculate_weights(train_data))
        #self.weighted_vote(result.values()[0], result.keys()[0], weigths)
        #self.weighted_vote(result.values()[1], result.keys()[1], weigths)

    def majority_vote(self, predicted_targets, actual_target):
        current_index = 0
        chunk_size = constants.SAMPLES_PER_TEST_SESSION
        print "Target expected: " + actual_target
        results= []
        while current_index + chunk_size < len(predicted_targets):
            chunk = predicted_targets[current_index: (current_index + chunk_size)]
            accuracy=len(chunk[chunk==actual_target])/float(chunk_size) * 100
            results.append(accuracy)
            print "Accuracy "+str(accuracy)
            current_index += chunk_size

        return results

    def vote_with_weights(self, predicted_targets, target_weights):
        selected_target = None
        max_accuracy = 0
        for target in target_weights:
            accuracy=predicted_targets.tolist().count(target)/float(len(predicted_targets))
            accuracy_with_weight_applied= accuracy*target_weights[target]
            if accuracy_with_weight_applied > max_accuracy:
                selected_target = target
                max_accuracy = accuracy_with_weight_applied
        return selected_target


    def calculate_weights(self, training_data):
        chunks = 3
        training_chunks = 2
        data_size = len(training_data)
        chunk_size = int(floor(data_size/chunks))
        extra_elements= data_size % chunk_size
        dataset_end = data_size-extra_elements
        chunk_borders = range(0,data_size,chunk_size)
        chunk_borders.append(dataset_end)

        weights_from_chunks = []

        for i in range(chunks):
            current_chunk = np.concatenate([training_data[:chunk_borders[i]] ,training_data[chunk_borders[i+1]:],training_data[chunk_borders[i]:chunk_borders[i+1]]])

            train_data, test_data = self.split_data(current_chunk, 0, training_chunks*chunk_size)
            prediction_result = self.train_and_test(train_data, test_data)

            weights = self.find_best_weights_for_target(prediction_result)
            weights_from_chunks.append(weights)

        print "Weights from chunks" + str(weights_from_chunks)
        avg_weights = {}

        for i,val in enumerate(weights_from_chunks):
            weight = weights_from_chunks[i]
            for target in weight:
                avg_weights[target] = (avg_weights.get(target, weight[target]) + weight[target])/2.0

        return avg_weights

    def find_best_weights_for_target(self, prediction_result):
        weights_to_test = [(x * 0.01, 200 * 0.01 - x * 0.01) for x in range(0, 200, 10)]
        max_accuracy=0
        best_weights = None
        for w in weights_to_test:
            weights = { prediction_result.keys()[0]: w[0], prediction_result.keys()[1]: w[1]}
            acc = self.find_accuracy_for_weights(prediction_result, weights)
            if acc>max_accuracy:
                max_accuracy = acc
                best_weights = weights

        return best_weights

    def find_accuracy_for_weights(self, prediction_result, weights):
        accuracies = []

        for target in weights:
            # TODO: should I split on samples instead of seconds ?
            second_chunks = self.split_on_seconds(prediction_result[target])
            single_target_voting_results=[]
            for chunk in second_chunks:
                single_target_voting_results.append(self.vote_with_weights(chunk, weights))
            accuracies.append(single_target_voting_results.count(target) / float(len(single_target_voting_results)))

        return sum(accuracies)/float(len(accuracies))

    def train_and_test(self, train_data, test_data):
        test_data = self.balance_data(test_data)
        learner = RFTLearner()
        learner.setup_classifier(train_data)
        actual_targets = test_data[:, 0]
        predicted_targets = learner.predict_samples(test_data)

        t1_target = constants.TARGETS.values()[0]
        t2_target = constants.TARGETS.values()[1]

        t1_predicted_targets = predicted_targets[actual_targets == t1_target]
        t2_predicted_targets = predicted_targets[actual_targets == t2_target]

        return {t1_target: t1_predicted_targets, t2_target: t2_predicted_targets}


    def split_on_seconds(self, data):
        chunks=[]
        current_index=0
        chunk_size=constants.SAMPLES_PER_TEST_SESSION
        while current_index + chunk_size < len(data):
            chunks.append(data[current_index: (current_index + chunk_size)])
            current_index += chunk_size
        return chunks
