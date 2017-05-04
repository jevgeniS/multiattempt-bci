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
from analyzing.voting.WeightsCalculator import WeightsCalculator
from analyzing.voting.WeightsVotingHandler import WeightsVotingHandler
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
        print "Majority Vote average accuracy between sessions: " + str(accuracy_from_sessions)
        print

        result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
        intra_juror_results = MajorityVotingHandler().vote(result)
        inter_juror_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES).vote(intra_juror_results)
        accuracy_from_sessions = AccuracyCalculator().get_accuracy(inter_juror_results)
        print "Majority Vote multiple jurors average accuracy: "+str(accuracy_from_sessions)
        print

        result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
        intra_juror_results = WeightsVotingHandler().vote_with_weights(result, self.weights)
        inter_juror_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES).vote(intra_juror_results)
        accuracy_from_sessions = AccuracyCalculator().get_accuracy(inter_juror_results)
        print "Weighted Vote multiple jurors average accuracy: " + str(accuracy_from_sessions)
        print

        result = ProbabilitiesToClassesConverter(self.thresholds).convert(probabilities)
        inter_sessions_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES * constants.SAMPLES_PER_TEST_SESSION).vote(result)
        accuracy_from_sessions = AccuracyCalculator().get_accuracy(inter_sessions_results)
        print "Vote with thresholds average accuracy between sessions: " + str(accuracy_from_sessions)
        print

        result = ProbabilitiesToClassesConverter(self.thresholds).convert(probabilities)
        intra_juror_results = MajorityVotingHandler().vote(result)
        inter_juror_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES).vote(intra_juror_results)
        accuracy_from_sessions = AccuracyCalculator().get_accuracy(inter_juror_results)
        print "Vote with thresholds multiple jurors average accuracy: "+str(accuracy_from_sessions)



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

        majority_accuracy = AccuracyCalculator().get_accuracy(result)
        print "Majority Vote average accuracy: " + str(majority_accuracy)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(majority_accuracy, required_p))
        ContingencyTableBuilder().build(result)
        print

        single_juror_avg_accuracy =AccuracyCalculator().get_accuracy(MajorityVotingHandler().vote(result))
        print "Majority Vote single juror average accuracy: " + str(single_juror_avg_accuracy)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(single_juror_avg_accuracy, required_p))
        ContingencyTableBuilder().build(MajorityVotingHandler().vote(result))
        print
        self.weights = WeightsCalculator().calculate_weights(probabilities)

        print "Selected weights"+str(self.weights)

        vote_result=WeightsVotingHandler().vote_with_weights(result, self.weights)
        single_juror_avg_accuracy = AccuracyCalculator().get_accuracy(vote_result)

        print "Weighted Vote single juror average accuracy: " + str(single_juror_avg_accuracy)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(single_juror_avg_accuracy, required_p))
        ContingencyTableBuilder().build(vote_result)
        print
        self.thresholds = ProbabilityThresholdCalculator().calculate_thresholds(probabilities)

        print "Selected thresholds" + str(self.thresholds)

        thr_vote_result = ProbabilitiesToClassesConverter(self.thresholds).convert(probabilities)
        thr_vote_res_accuracy = AccuracyCalculator().get_accuracy(thr_vote_result)

        print "Vote with thresholds average accuracy: " + str(thr_vote_res_accuracy)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(thr_vote_res_accuracy, required_p))
        ContingencyTableBuilder().build(thr_vote_result)
        print

        single_juror_avg_accuracy = AccuracyCalculator().get_accuracy(MajorityVotingHandler().vote(thr_vote_result))
        print "Vote with thresholds single juror average accuracy: " + str(single_juror_avg_accuracy)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(single_juror_avg_accuracy, required_p))
        ContingencyTableBuilder().build(MajorityVotingHandler().vote(result))
        print
