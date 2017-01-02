from src.analyzing.RFTLearner import RFTLearner
from src.util.DataStorer import DataStorer


class TrainingService(object):
    def start(self):
        data = DataStorer.read()
        n = 10
        count = 0
        for i in range(n):
            count += RFTLearner().setup_classifier(data)

        print count / n