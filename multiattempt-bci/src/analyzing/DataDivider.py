from constants import constants
import numpy as np

class DataDivider(object):
    def divide_on_two(self, data):
        training_border_index = int(constants.TRAINING_DATA_PERCENTAGE / 100.0 * len(data))
        train_data, test_data = self.split_data(data, 0, training_border_index)
        test_data = self.balance_data(test_data)
        return train_data, test_data

    def split_data(self, data, training_start_index, training_stop_index):
        data=np.array(data)
        training_data = np.array(data[training_start_index:training_stop_index])
        test_data = np.concatenate((data[:training_start_index], data[training_stop_index:]))
        return training_data, test_data

    def balance_data(self, data):
        targets = data[:, 0]
        targets1 = [i for i, val in enumerate(targets) if val == constants.TARGETS.values()[0]]
        targets2 = [i for i, val in enumerate(targets) if val == constants.TARGETS.values()[1]]
        t1_len = len(targets1)
        t2_len = len(targets2)
        if t1_len < t2_len:
            return np.concatenate((data[targets1], data[targets2[:t1_len]]))
        else:
            return np.concatenate((data[targets1[:t2_len]], data[targets2]))

    def split_on_sessions(self, data, chunk_size):
        chunks = []
        current_index = 0
        while current_index + chunk_size <= len(data):
            chunks.append(data[current_index: (current_index + chunk_size)])
            current_index += chunk_size
        return chunks

    def split_data_percent(self, data, test_data_start_percent):

        test_data_start = int(test_data_start_percent/100.0*len(data))
        test_data_percent = 100 - int(constants.TRAINING_DATA_PERCENTAGE)
        test_data_len = int(test_data_percent/100.0*len(data))
        test_data_end = test_data_start+test_data_len
        test_data = np.array(data[test_data_start:test_data_end])
        training_data = np.array(data[:test_data_start] + data[test_data_end:])

        return training_data, test_data