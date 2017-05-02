from itertools import groupby

from analyzing.DataDivider import DataDivider
from constants import constants


class MajorityVotingHandler(object):

    def __init__(self, chunk_size=None):
        if chunk_size is None:
            self.chunk_size= constants.SAMPLES_PER_TEST_SESSION
        else:
            self.chunk_size = chunk_size

    def vote(self, result):
        new_result={}
        for key in result:
            new_result[key] = self.majority_vote(result[key])
        return new_result

    def majority_vote(self, predicted_probs):
        results = []
        sessions = DataDivider().split_on_sessions(predicted_probs, self.chunk_size)
        for session_data in sessions:
            results.append(self.vote_without_splitting_on_sessions(session_data))

        return results

    def vote_without_splitting_on_sessions(self, predicted_probs):
        max_size= 0
        max_target= None
        for key, group in groupby(predicted_probs):
            size= len(list(group))
            if size> max_size:
                max_size= size
                max_target= key

        return max_target
