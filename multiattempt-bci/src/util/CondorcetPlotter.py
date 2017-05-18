from itertools import cycle
from random import shuffle

from analyzing.CondorcetCalculator import CondorcetCalculator
import matplotlib.pyplot as plt


class CondorcetPlotter(object):

    def plot(self, baselines, maximum):
        legend =[]
        lines = ["-", "--", "-.", ":"]
        shuffle(lines)
        linecycler = cycle(lines)
        for method_name in baselines:
            voters=[]
            accs = []
            i=1
            acc=0
            while acc<maximum:
                acc=CondorcetCalculator().calculate_prob_for_voters(baselines[method_name], i)[1]
                voters.append(i)
                accs.append(acc)
                i += 2
            legend.append(method_name)
            plt.plot(voters, accs, next(linecycler), linewidth=2.0)
        plt.legend(legend, loc='lower right')
        plt.xlabel('Number of trials', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.show()
