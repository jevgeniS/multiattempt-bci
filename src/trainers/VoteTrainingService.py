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
from util.CondorcetPlotter import CondorcetPlotter


class VoteTrainingService(TrainingService):
    weights=None
    decision_threshold_x_proba_threshold=None
    thresholds = None
    learner=None

    def __init__(self, train_data_storer, cross_validator=None):
        self.train_data_storer=train_data_storer
        self.cross_validator = cross_validator

    def test(self, test_data):
        print "____TESTING_RESULTS____"
        probabilities = self.learner.predict_samples(test_data)

        name = "No voting 1/2-probability threshold"
        result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
        single_attempt_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES*constants.SAMPLES_PER_TEST_SESSION).vote(result)
        accuracy = AccuracyCalculator().get_accuracy(single_attempt_results)
        print name+": " + str(accuracy)
        print

        name = "No voting T-probability threshold"
        result = ProbabilitiesToClassesConverter(self.thresholds).convert(probabilities)
        single_attempt_results = MajorityVotingHandler(
            constants.USER_TEST_SAMPLES * constants.SAMPLES_PER_TEST_SESSION).vote(result)
        accuracy = AccuracyCalculator().get_accuracy(single_attempt_results)
        print name + ": " + str(accuracy)
        print

        name = "Majority voting 1/2-probability threshold"
        result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
        single_attempt_results = MajorityVotingHandler().vote(result)
        multiple_attempt_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES).vote(single_attempt_results)
        ContingencyTableBuilder().build(multiple_attempt_results)
        accuracy = AccuracyCalculator().get_accuracy(multiple_attempt_results)
        print name+": "+str(accuracy)
        print

        name = "Majority voting T-probability threshold"
        result = ProbabilitiesToClassesConverter(self.thresholds).convert(probabilities)
        single_attempt_results = MajorityVotingHandler().vote(result)
        multiple_attempt_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES).vote(single_attempt_results)
        ContingencyTableBuilder().build(multiple_attempt_results)
        accuracy= AccuracyCalculator().get_accuracy(multiple_attempt_results)
        print name + ": " + str(accuracy)
        print

        name = "Decision threshold voting 1/2-probability threshold"
        result = MajorityProbabilitiesToClassesConverter().convert(probabilities)
        single_attempt_results = WeightsVotingHandler().vote_with_weights(result, self.weights)
        multiple_attempt_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES).vote(single_attempt_results)
        ContingencyTableBuilder().build(multiple_attempt_results)
        accuracy = AccuracyCalculator().get_accuracy(multiple_attempt_results)
        print name+": " + str(accuracy)
        print

        name = "Decision threshold voting T-probability threshold"
        result = ProbabilitiesToClassesConverter(self.thresholds).convert(probabilities)
        single_attempt_results = WeightsVotingHandler().vote_with_weights(result, self.weights)
        multiple_attempt_results = MajorityVotingHandler(constants.USER_TEST_SAMPLES).vote(single_attempt_results)
        ContingencyTableBuilder().build(multiple_attempt_results)
        accuracy = AccuracyCalculator().get_accuracy(multiple_attempt_results)
        print name + ": " + str(accuracy)
        print


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
        condorcet_plotter_data={}

        acc = AccuracyCalculator().get_accuracy(result)
        name="No vote 1/2-probability threshold"
        print name+": " + str(acc)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(acc, required_p))
        condorcet_plotter_data[name]=acc
        ContingencyTableBuilder().build(result)
        print

        self.thresholds = ProbabilityThresholdCalculator().calculate_thresholds(probabilities)
        print "Selected thresholds" + str(self.thresholds)
        prob_thr_vote_result = ProbabilitiesToClassesConverter(self.thresholds).convert(probabilities)

        acc = AccuracyCalculator().get_accuracy(prob_thr_vote_result)
        name = "No vote T-probability threshold"
        print name + ": " + str(acc)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(acc, required_p))
        condorcet_plotter_data[name] = acc
        ContingencyTableBuilder().build(prob_thr_vote_result)
        print

        acc = AccuracyCalculator().get_accuracy(MajorityVotingHandler().vote(result))
        name ="Majority voting 1/2-probability threshold"
        print name+": " + str(acc)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(acc, required_p))
        condorcet_plotter_data[name] = acc
        ContingencyTableBuilder().build(MajorityVotingHandler().vote(result))
        print

        name = "Majority voting T-probability threshold"
        thr_vote_result = MajorityVotingHandler().vote(prob_thr_vote_result)
        acc = AccuracyCalculator().get_accuracy(thr_vote_result)
        print name + ": " + str(acc)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(acc, required_p))
        condorcet_plotter_data[name] = acc
        ContingencyTableBuilder().build(thr_vote_result)
        print

        self.weights = WeightsCalculator().calculate_weights(probabilities)
        print "Selected decision thresholds"+str(self.weights)

        vote_result=WeightsVotingHandler().vote_with_weights(result, self.weights)
        acc = AccuracyCalculator().get_accuracy(vote_result)

        name="Decision threshold voting 1/2-probability threshold"
        print name+": " + str(acc)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(acc, required_p))
        condorcet_plotter_data[name] = acc
        ContingencyTableBuilder().build(vote_result)
        print

        vote_result = WeightsVotingHandler().vote_with_weights(prob_thr_vote_result, self.weights)
        acc = AccuracyCalculator().get_accuracy(vote_result)

        name = "Decision threshold voting T-probability threshold"
        print name + ": " + str(acc)
        print "Samples required for p>=" + str(required_p) + " " + str(
            CondorcetCalculator().calculate_number_of_voters(acc, required_p))
        condorcet_plotter_data[name] = acc
        ContingencyTableBuilder().build(vote_result)
        print

        #CondorcetPlotter().plot(condorcet_plotter_data, 0.99)
