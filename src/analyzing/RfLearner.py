from sklearn.ensemble import RandomForestClassifier

from constants import constants


class RfLearner:

    def __init__(self):
        self.classifier = None
        self.accuracy = 0.0


    def predict(self, sample):
        return self.classifier.predict(sample)


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
        test_features = test_data[:, 1:]
        prediction_result=self.classifier.predict_proba(test_features)
        result={}
        for i,cls in enumerate(self.classifier.classes_):
            result[cls]=prediction_result[:, i]
        return result




