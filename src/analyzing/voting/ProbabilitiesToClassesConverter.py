class ProbabilitiesToClassesConverter(object):

    def __init__(self, thresholds):
        self.thresholds = thresholds

    def convert(self, result):
        new_result={}
        for key in result:
            new_result[key] = self.prob_threshold_vote(key, result[key])
        return new_result

    def prob_threshold_vote(self, cls, predicted_probs):
        threshold= self.thresholds[cls]
        results = []
        for prob in predicted_probs:
            if prob>threshold:
                result = cls
            else:
                other_cls = [c for c in self.thresholds.keys() if c!=cls][0]
                result = other_cls
            results.append(result)

        return results
