import numpy as np

from TrainingService import TrainingService
from analyzing.DataDivider import DataDivider
from analyzing.voting.MajorityProbabilitiesToClassesConverter import MajorityProbabilitiesToClassesConverter


class CrossValidator(object):

    def validate(self, data):
        parts_n = 5
        part_elements_n = int(len(data) / parts_n)
        prediction_result={}
        for i in range(parts_n):
            test_data_chunk_start = i * part_elements_n
            test_data_chunk_end = test_data_chunk_start + part_elements_n

            training_data_1 = data[:test_data_chunk_start]
            training_data_2 = data[test_data_chunk_end:]
            training_data = np.concatenate([training_data_1, training_data_2], axis=0)
            test_data = data[test_data_chunk_start:test_data_chunk_end]
            #test_data = DataDivider().balance_data(test_data)
            probabilities, clf= TrainingService().train_and_test(training_data, test_data)

            for t in probabilities:
                pr = prediction_result.get(t, [])
                pr.extend(probabilities[t])
                prediction_result[t] = pr

            #MajorityVotingHandler().vote()

        #ContingencyTableBuilder().build(prediction_result)

        return prediction_result
