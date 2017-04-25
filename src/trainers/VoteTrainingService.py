from math import floor

import numpy as np
import operator

from TrainingService import TrainingService
from analyzing.RfLearner import RfLearner
from constants import constants
from trainers.CondorcetCalculator import CondorcetCalculator


class VoteTrainingService(TrainingService):
    weights=None
    classificator=None

    def __init__(self, train_data_storer):
        self.train_data_storer=train_data_storer

    def test(self, test_data):
        result = self.classificator.predict(test_data)
        mv_result = self.majority_vote_test(result)
        print "Majority vote result: "+str(mv_result)
        wv_result = self.weighted_vote_test(result, self.weights)
        print "Weighted vote result: "+str(wv_result)
        return mv_result, wv_result


    def train(self):
        data = self.train_data_storer.read()
        w_acc, m_acc = self.train_with_weighted_vote(data)

        print "Majority Vote: "
        print "Total accuracy: " + str(m_acc)
        required_p= 0.99
        condorcet_result=CondorcetCalculator.calculate_number_of_voters(m_acc, required_p)
        print "Samples required for p>="+str(required_p)+" "+str(condorcet_result)
        print
        print "Weighted Vote: "
        print "Total accuracy: " + str(w_acc)
        condorcet_result=CondorcetCalculator.calculate_number_of_voters(w_acc, required_p)
        print "Samples required for p>=" + str(required_p) + " " + str(condorcet_result)


    def train_with_weighted_vote(self, data):
        training_border_index = int(constants.TRAINING_DATA_PERCENTAGE / 100.0 * len(data))
        train_data, test_data = self.split_data(data, 0, training_border_index)
        result = self.train_and_test(train_data, test_data, True)
        m_acc1 = self.majority_vote(result.values()[0], result.keys()[0])
        m_acc2 = self.majority_vote(result.values()[1], result.keys()[1])
        majority_accuracy = ((m_acc1 + m_acc2) / 2.0)
        print "Majority Vote accuracy: " + str(majority_accuracy)
        self.weights = self.calculate_weights(train_data)

        print(self.weights)

        acc1 = self.weighted_vote(result.values()[0], result.keys()[0], self.weights)
        acc2 = self.weighted_vote(result.values()[1], result.keys()[1], self.weights)
        total_accuracy = ((acc1 + acc2) / 2.0)
        print "Weighted Vote accuracy: " + str(total_accuracy)
        return total_accuracy, majority_accuracy

    def majority_vote(self, predicted_targets, actual_target):
        # print "Target expected: " + actual_target
        results = []
        sessions = self.split_on_sessions(predicted_targets)
        for session_data in sessions:
            accuracy = len(session_data[session_data == actual_target]) / float(len(session_data))
            results.append(accuracy)
            # print "Accuracy "+str(accuracy)
        return sum(results) / float(len(results))

    def majority_vote_test(self, predicted_targets):
        sessions = self.split_on_sessions(predicted_targets)

        vote_results = {}
        for session_data in sessions:
            for t in constants.TARGETS.values():
                p = session_data.tolist().count(t) / float(len(session_data))
                vote_results[t] = (vote_results.get(t, p) + p)/2.0

        max_key = max(vote_results.iteritems(), key=operator.itemgetter(1))[0]

        return (max_key, vote_results[max_key])

    def weighted_vote(self, predicted_targets, actual_target, weights):
        # print "Target expected: " + actual_target
        results = []
        sessions = self.split_on_sessions(predicted_targets)
        for session_data in sessions:
            results.append(self.vote_with_weights(session_data, weights) == actual_target)

            # accuracy=len(chunk[chunk==actual_target])/float(chunk_size) * 100
            # results.append(accuracy)

        accuracy = results.count(True) / float(len(results))
        # print "Accuracy "+str(accuracy)

        return accuracy

    def weighted_vote_test(self, predicted_targets, weights):
        results = []
        sessions = self.split_on_sessions(predicted_targets)
        for session_data in sessions:
            results.append(self.vote_with_weights(session_data, weights))

        vote_results={}
        for t in constants.TARGETS.values():
            p = results.count(t) / float(len(results))
            vote_results[t] = p

        max_key= max(vote_results.iteritems(), key=operator.itemgetter(1))[0]

        return (max_key,vote_results[max_key])


    def vote_with_weights_coef(self, predicted_targets, target_weights):
        selected_target = None
        max_accuracy = 0
        for target in target_weights:
            accuracy = predicted_targets.tolist().count(target) / float(len(predicted_targets))
            accuracy_with_weight_applied = accuracy * target_weights[target]
            if accuracy_with_weight_applied > max_accuracy:
                selected_target = target
                max_accuracy = accuracy_with_weight_applied
        return selected_target

    def vote_with_weights(self, predicted_targets, target_weights):
        target = target_weights.keys()[0]
        matches = predicted_targets.tolist().count(target)
        whole_number_treshold = round(target_weights[target])
        if matches>=whole_number_treshold:
            return target

        return target_weights.keys()[1]

    def calculate_weights(self, training_data):
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

            train_data, test_data = self.split_data(current_chunk, 0, training_chunks * chunk_size)
            prediction_result = self.train_and_test(train_data, test_data)

            weights = self.find_best_weights_for_target(prediction_result)
            weights_from_chunks.append(weights)

        print "Weights from chunks" + str(weights_from_chunks)
        avg_weights = {}

        for i, val in enumerate(weights_from_chunks):
            weight = weights_from_chunks[i]
            for target in weight:
                avg_weights[target] = (avg_weights.get(target, weight[target]) + weight[target]) / 2.0

        return avg_weights

    def find_best_weights_for_target(self, prediction_result):
        number_of_samples = constants.SAMPLES_PER_TEST_SESSION
        weights_to_test = [(x, number_of_samples - x) for x in range(number_of_samples)]
        #weights_to_test = [(x * 0.01, 200 * 0.01 - x * 0.01) for x in range(30, 170, 1)]
        max_accuracy = 0
        best_weights = None
        for w in weights_to_test:
            weights = {prediction_result.keys()[0]: w[0], prediction_result.keys()[1]: w[1]}
            acc = self.find_accuracy_for_weights(prediction_result, weights)
            if acc > max_accuracy:
                max_accuracy = acc
                best_weights = weights

        return best_weights

    def find_accuracy_for_weights(self, prediction_result, weights):
        accuracies = []

        for target in weights:
            sessions = self.split_on_sessions(prediction_result[target])
            single_target_voting_results = []
            for session_data in sessions:
                single_target_voting_results.append(self.vote_with_weights(session_data, weights))
            accuracies.append(single_target_voting_results.count(target) / float(len(single_target_voting_results)))

        return sum(accuracies) / float(len(accuracies))


    def train_and_test(self, train_data, test_data, set_classificator=False):

        test_data = self.balance_data(test_data)

        learner = RfLearner()
        learner.setup_classifier(train_data)
        if set_classificator:
            self.classificator=learner
        actual_targets = test_data[:, 0]
        predicted_targets = learner.predict_samples(test_data)


        t1_target = constants.TARGETS.values()[0]
        t2_target = constants.TARGETS.values()[1]

        t1_predicted_targets = predicted_targets[actual_targets == t1_target]
        t2_predicted_targets = predicted_targets[actual_targets == t2_target]

        return {t1_target: t1_predicted_targets, t2_target: t2_predicted_targets}

    def split_on_sessions(self, data):
        chunks = []
        current_index = 0
        chunk_size = constants.SAMPLES_PER_TEST_SESSION
        while current_index + chunk_size <= len(data):
            chunks.append(data[current_index: (current_index + chunk_size)])
            current_index += chunk_size
        return chunks
