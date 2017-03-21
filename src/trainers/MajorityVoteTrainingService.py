from collections import Counter
from math import floor

from src import constants
from src.TrainingService import TrainingService
from src.analyzing.RFTLearner import RFTLearner
from src.util.DataStorer import DataStorer
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
        print(self.calculate_threshold(train_data))
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

    def vote_with_threshold(self, predicted_targets, high_probability_target, threshold):
        most_common_targets=Counter(predicted_targets).most_common()
        accuracy = most_common_targets[0][1] / float(len(predicted_targets))
        if most_common_targets[0][0]==high_probability_target:
            if accuracy>=threshold:
                return most_common_targets[0][0]

        return most_common_targets[1][0]


    def calculate_threshold(self, training_data):
        chunks = 3
        training_chunks = 2
        data_size = len(training_data)
        chunk_size = int(floor(data_size/chunks))
        extra_elements= data_size % chunk_size
        dataset_end = data_size-extra_elements
        chunk_borders = range(0,data_size,chunk_size)
        chunk_borders.append(dataset_end)

        thresholds_from_chunks = {}
        high_probability_targets_from_chunks = []

        for i in range(chunks):
            current_chunk = np.concatenate([training_data[:chunk_borders[i]] ,training_data[chunk_borders[i+1]:],training_data[chunk_borders[i]:chunk_borders[i+1]]])

            train_data, test_data = self.split_data(current_chunk, 0, training_chunks*chunk_size)
            result = self.train_and_test(train_data, test_data)

            t1 = result.keys()[0]
            t2 = result.keys()[1]

            t1_results = self.majority_vote(result.values()[0], t1)
            t1_results_avg = sum(t1_results)/len(t1_results)
            t2_results = self.majority_vote(result.values()[1], t2)
            t2_results_avg = sum(t2_results) / len(t2_results)

            if t1_results_avg>t2_results_avg:
                high_probability_targets_from_chunks.append(t1)
            else:
                high_probability_targets_from_chunks.append(t2)

            treshold1 = self.find_best_threshold_for_high_accuracy_target(t1, result)
            tresholds1 = thresholds_from_chunks.get(t1, [])
            tresholds1.append(treshold1)
            thresholds_from_chunks[t1] =tresholds1

            treshold2 = self.find_best_threshold_for_high_accuracy_target(t2, result)
            tresholds2 = thresholds_from_chunks.get(t2, [])
            tresholds2.append(treshold2)
            thresholds_from_chunks[t2] = tresholds2


        if len(list(set(high_probability_targets_from_chunks)))>1:
            #raise Exception("High probability targets differ within chunks")
            most_common_targets = Counter(high_probability_targets_from_chunks).most_common()
            high_probability_target = most_common_targets[0][0]

        else:
            high_probability_target = high_probability_targets_from_chunks[0]

        print thresholds_from_chunks[high_probability_target]

        avg_threshold= sum(thresholds_from_chunks[high_probability_target])/float(len(thresholds_from_chunks[high_probability_target]))

        return {high_probability_target:avg_threshold}

    def find_best_threshold_for_high_accuracy_target(self, high_probability_target, result):
        thresholds= [x * 0.01 for x in range(50,100,5)]
        max_accuracy=0
        threshold=0
        for t in thresholds:
            acc=self.find_accuracy_for_threshold(result, high_probability_target, t)
            if acc>max_accuracy:
                max_accuracy = acc
                threshold = t
        return threshold

    def find_accuracy_for_threshold(self, result, high_probability_target, threshold):
        second_chunks = self.split_on_seconds(result[high_probability_target])
        prediction_result=[]
        for chunk in second_chunks:
            prediction_result.append(self.vote_with_threshold(chunk, high_probability_target, threshold))
        accuracy = prediction_result.count(high_probability_target) / float(len(prediction_result))

        return accuracy

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
