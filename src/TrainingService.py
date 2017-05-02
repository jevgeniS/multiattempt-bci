from analyzing.RfLearner import RfLearner


class TrainingService(object):

    def train_and_test(self, train_data, test_data):

        learner = RfLearner()
        learner.setup_classifier(train_data)
        #actual_targets = test_data[:, 0]
        result = learner.predict_samples(test_data)

        return result,learner
