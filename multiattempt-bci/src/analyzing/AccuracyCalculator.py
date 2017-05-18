class AccuracyCalculator(object):

    def get_accuracy(self, result):
        length = len(result.values()[0])+len(result.values()[1])
        if length == 0:
            return 1.0

        matches1 = self.get_matches(result.values()[0], result.keys()[0])
        matches2 = self.get_matches(result.values()[1], result.keys()[1])
        accuracy = (matches1+matches2) / float(length)

        return accuracy

    def get_matches(self, predicted_targets, actual_target):
        matches = len([i for i,x in enumerate(predicted_targets) if x == actual_target])
        return matches
