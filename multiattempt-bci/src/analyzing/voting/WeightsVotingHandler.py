from analyzing.DataDivider import DataDivider
from constants import constants


class WeightsVotingHandler(object):

    def __init__(self, chunk_size=None):
        if chunk_size is None:
            self.chunk_size= constants.SAMPLES_PER_TEST_SESSION
        else:
            self.chunk_size = chunk_size

    def vote_with_weights(self, result, weights):
        new_result={}
        for key in result:
            new_result[key] = self.weighted_vote(result[key], weights)
        return new_result

    def weighted_vote(self, predicted_targets, weights):
        results = []
        sessions = DataDivider().split_on_sessions(predicted_targets, self.chunk_size)
        for session_data in sessions:
            results.append(self.vote_with_weights1(session_data, weights))

        return results

    def vote_with_weights1(self, predicted_targets, target_weights):
        target = target_weights.keys()[0]
        matches = len([i for i in predicted_targets if i == target])
        whole_number_threshold = round(target_weights[target])
        if matches>=whole_number_threshold:
            return target

        return target_weights.keys()[1]

