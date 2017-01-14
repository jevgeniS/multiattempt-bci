import numpy as np
from datetime import datetime

import winsound

from LearningService import LearningService
from src.TestingService import TestingService
from src.TrainingService import TrainingService
from src.analyzing.RFTLearner import RFTLearner
from src.processing.Plotter import Plotter
from src.util.DataStorer import DataStorer


def test():
    TestingService().start()


def train():
    training_start = datetime(2017, 1, 4, 21, 45)
    test_start1 = datetime(2017, 1, 7, 14, 14)
    test_start2 = datetime(2018, 1, 4, 22, 6)
    training_data = DataStorer.read_interval(training_start, test_start1)
    test_data1 = DataStorer.read_interval(test_start1, test_start2)
    test_data2 = DataStorer.read_interval(test_start2, datetime.now())

    #data = DataStorer.read()
    n = 100
    count1 = 0
    count2 = 0
    rf = RFTLearner()

    #training_data, test_data = rf.split_data(DataStorer.read_interval(datetime.min, training_start))
    for i in range(n):
        rf.setup_classifier(training_data)
        accuracy1 = rf.get_accuracy(test_data1)
        #accuracy2 = rf.get_accuracy(test_data2)
        print "Accuracy 1: " + str(accuracy1)
        #print "Accuracy 2: " + str(accuracy2)
        count1 += accuracy1
        #count2 += accuracy2

    print "Average accuracy 1: " + str(count1 / n)
    print "Average accuracy 2: " + str(count2 / n)



def learn():
    i = 1
    while (True):
        print "Try number:"+str(i)
        LearningService().start()
        i += 1


def plot():
    source_data = np.array(DataStorer().read())
    target = 'Left'
    data = source_data[np.where(source_data[:, 0] == target)]
    Plotter(target).plot_avg(data)



if __name__ == "__main__":
    plot()
    exit()
    answer = raw_input("If you want to continue with learning mode please type '1', with training mode '2'")
    if answer == "1":
        print "Switched to learning mode"
        learn()
    if answer == "2":
        print "Switched to training mode"
        #train()
        TrainingService().start()
    else:
        print "Switched to test mode"


