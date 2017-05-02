import numpy as np

import constants
from LearningService import LearningService
from TestingService import TestingService
from analyzing.CrossValidator import CrossValidator
from processing.Plotter import Plotter
from trainers.VoteTrainingService import VoteTrainingService
from util.DataStorer import DataStorer


def plot():
    source_data = np.array(DataStorer().read())
    target = 'Left'
    data = source_data[np.where(source_data[:, 0] == target)]
    Plotter(target).plot_avg(data)


if __name__ == "__main__":
    modes = constants.constants.APP_MODES
    answer = raw_input("Choose mode: \n"+str(modes)+"\n")
    if answer in constants.constants.APP_MODES.keys():
        print "Switched to "+constants.constants.APP_MODES[answer]
        if answer == "1":
            LearningService().measure()
        if answer == "2":
            VoteTrainingService(DataStorer(constants.constants.DATA_FILE), CrossValidator()).train()
        if answer == "3":
            TestingService(DataStorer(constants.constants.DATA_FILE)).measure()
        if answer == "4":
            TestingService(DataStorer(constants.constants.DATA_FILE)).calculate_target()
    else:
        print "No such kind of mode"


