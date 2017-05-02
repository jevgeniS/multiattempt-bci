from analyzing.RfLearner import RfLearner


class TrainingService(object):


    def train_and_test(self, train_data, test_data):

        learner = RfLearner()
        learner.setup_classifier(train_data)
        actual_targets = test_data[:, 0]
        predicted_targets = learner.predict_samples(test_data)

        result = {}
        for target in predicted_targets:
            result[target] = []
            for i,at in enumerate(actual_targets):
                if at==target:
                    result[target].append(predicted_targets[target][i])


        return result,learner
