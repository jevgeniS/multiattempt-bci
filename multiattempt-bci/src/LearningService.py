from BaseMeasuringService import BaseMeasuringService
from util.TimeStampGenerator import get_timestamp


class LearningService(BaseMeasuringService):

    def get_number_of_samples_per_session(self):
        return 1

    def process_prediction_data(self, samples):
        answer = raw_input("Type X do skip save of current try")

        if answer != 'X':
            self.train_data_storer.store(samples, get_timestamp())





