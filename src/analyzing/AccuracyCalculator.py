class AccuracyCalculator(object):

    def get_accuracy(self, result):

        t1_acc = self.get_avg_accuracy(result.values()[0], result.keys()[0])
        t2_acc = self.get_avg_accuracy(result.values()[1], result.keys()[1])
        accuracy = ((t1_acc + t2_acc) / 2.0)

        return accuracy

    def get_avg_accuracy(self, predicted_targets, actual_target):
        matches = len([i for i,x in enumerate(predicted_targets) if x == actual_target])
        accuracy= matches/float(len(predicted_targets))
        return accuracy
