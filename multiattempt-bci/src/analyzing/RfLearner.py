from sklearn.ensemble import RandomForestClassifier

from constants import constants


class RfLearner:

    def __init__(self):
        self.classifier = None
        self.accuracy = 0.0


    #def predict(self, sample):
        #return self.classifier.predict(sample)


    def setup_classifier(self, training_data):
        self.classifier = self.train(training_data)

    def train(self, training_data):
        training_targets = training_data[:, 0]
        training_features = training_data[:, 1:]


        clf = RandomForestClassifier(n_estimators=constants.RF_CLF_TREES, n_jobs=2, random_state=0)
        #clf = DecisionTreeClassifier(random_state=0)
        clf=clf.fit(training_features, training_targets)

        return clf

    def predict_samples(self, test_data):
        actual_targets=test_data[:, 0]
        test_features = test_data[:, 1:]
        prediction_result=self.classifier.predict_proba(test_features)
        predicted_targets={}
        for i,cls in enumerate(self.classifier.classes_):
            predicted_targets[cls]=prediction_result[:, i]

        result = {}
        for target in predicted_targets:
            result[target] = []
            for i, at in enumerate(actual_targets):
                if at == target:
                    result[target].append(predicted_targets[target][i])

        return result




