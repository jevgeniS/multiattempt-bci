import operator

import numpy as np

from TrainingService import TrainingService
from analyzing.AccuracyCalculator import AccuracyCalculator
from analyzing.CondorcetCalculator import CondorcetCalculator
from analyzing.ContingencyTableBuilder import ContingencyTableBuilder
from analyzing.DataDivider import DataDivider
from analyzing.RfLearner import RfLearner
from analyzing.voting.MajorityProbabilitiesToClassesConverter import MajorityProbabilitiesToClassesConverter
from analyzing.voting.MajorityVotingHandler import MajorityVotingHandler
from analyzing.voting.ProbabilitiesToClassesConverter import ProbabilitiesToClassesConverter
from analyzing.voting.ProbabilityThresholdCalculator import ProbabilityThresholdCalculator
from analyzing.voting.WeightsVotingHandler import WeightsVotingHandler
from analyzing.voting.WeightsCalculator import WeightsCalculator
from constants import constants


class VoteTrainingService(TrainingService):
    weights=None
    thresholds = None
    learner=None

    def __init__(self, train_data_storer, cross_validator=None):
        self.train_data_storer=train_data_storer
        self.cross_validator = cross_validator

    def test(self, test_data):
        print "____TESTING_RESULTS____"
        probabilities = self.learner.predict_samples(test_data)

        result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
        inter_sessions_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES*constants.SAMPLES_PER_TEST_SESSION).vote(result)
        accuracy_from_sessions = AccuracyCalculator().get_accuracy(inter_sessions_results)
        print "Majority accuracy: " + str(accuracy_from_sessions)
        print

        result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
        intra_juror_results = MajorityVotingHandler().vote(result)
        inter_juror_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES).vote(intra_juror_results)
        accuracy_from_sessions = AccuracyCalculator().get_accuracy(inter_juror_results)
        print "Majority accuracy between sessions: "+str(accuracy_from_sessions)
        print

        result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
        intra_juror_results = WeightsVotingHandler().vote_with_weights(result, self.weights)
        inter_juror_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES).vote(intra_juror_results)
        accuracy_from_sessions = AccuracyCalculator().get_accuracy(inter_juror_results)
        print "Weighted Vote accuracy between sessions: " + str(accuracy_from_sessions)
        print

        result = ProbabilitiesToClassesConverter(self.thresholds).convert(probabilities)
        inter_sessions_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES * constants.SAMPLES_PER_TEST_SESSION).vote(result)
        accuracy_from_sessions = AccuracyCalculator().get_accuracy(inter_sessions_results)
        print "Vote with thresholds: " + str(accuracy_from_sessions)
        print

        result = ProbabilitiesToClassesConverter(self.thresholds).convert(probabilities)
        intra_juror_results = MajorityVotingHandler().vote(result)
        inter_juror_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES).vote(intra_juror_results)
        accuracy_from_sessions = AccuracyCalculator().get_accuracy(inter_juror_results)
        print "Vote with thresholds between sessions accuracy: "+str(accuracy_from_sessions)



    def train(self):
        data = self.train_data_storer.read()
        data = np.array(data)
        self.train_baselines(data)

    def train_baselines(self, data):
        required_p = constants.REQUIRED_PROBABILITY
        print "____TRAINING_RESULTS____"

        if self.cross_validator is None:
            train_data, test_data = DataDivider().divide_on_two(data)
            probabilities, clf = self.train_and_test(train_data, test_data)
            result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
            self.learner = clf
        else:
            probabilities = self.cross_validator.validate(data)
            result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
            self.learner = RfLearner()
            self.learner.setup_classifier(data)
            train_data = data

        #result = MajorityVotingHandler().vote(result)
        ContingencyTableBuilder().build(result)

        majority_accuracy = AccuracyCalculator().get_accuracy(result)
        print "Majority Vote accuracy: " + str(majority_accuracy)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(majority_accuracy, required_p))
        print

        majority_accuracy_between_sessions =AccuracyCalculator().get_accuracy(MajorityVotingHandler().vote(result))
        print "Majority Vote between sessions accuracy: " + str(majority_accuracy_between_sessions)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(majority_accuracy_between_sessions, required_p))
        print
        self.weights = WeightsCalculator().calculate_weights(probabilities)

        print "Selected weights"+str(self.weights)

        vote_result=WeightsVotingHandler().vote_with_weights(result, self.weights)
        vote_res_accuracy = AccuracyCalculator().get_accuracy(vote_result)

        print "Weighted Vote accuracy: " + str(vote_res_accuracy)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(vote_res_accuracy, required_p))
        print
        self.thresholds = ProbabilityThresholdCalculator().calculate_thresholds(probabilities)

        print "Selected thresholds" + str(self.thresholds)

        thr_vote_result = ProbabilitiesToClassesConverter(self.thresholds).convert(probabilities)
        thr_vote_res_accuracy = AccuracyCalculator().get_accuracy(thr_vote_result)

        print "Vote with thresholds accuracy: " + str(thr_vote_res_accuracy)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(thr_vote_res_accuracy, required_p))
        print

        thresholds_vote_accuracy_between_sessions = AccuracyCalculator().get_accuracy(MajorityVotingHandler().vote(thr_vote_result))
        print "Vote with thresholds between sessions accuracy: " + str(thresholds_vote_accuracy_between_sessions)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(thresholds_vote_accuracy_between_sessions, required_p))
        print

    def majority_vote_test(self, predicted_targets):
        sessions = DataDivider().split_on_sessions(predicted_targets)

        vote_results = {}
        for session_data in sessions:
            for t in constants.TARGETS.values():
                p = session_data.tolist().count(t) / float(len(session_data))
                vote_results[t] = (vote_results.get(t, p) + p)/2.0

        max_key = max(vote_results.iteritems(), key=operator.itemgetter(1))[0]

        return (max_key, vote_results[max_key])



    def weighted_vote_test(self, predicted_targets, weights):
        results = []
        sessions = DataDivider().split_on_sessions(predicted_targets)
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







