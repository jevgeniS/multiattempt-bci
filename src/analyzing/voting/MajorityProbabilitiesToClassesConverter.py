from analyzing.voting.ProbabilitiesToClassesConverter import ProbabilitiesToClassesConverter


class MajorityProbabilitiesToClassesConverter(ProbabilitiesToClassesConverter):

    def __init__(self):
        thresholds= {'Excitement': 0.5, 'Relax': 0.5}
        super(MajorityProbabilitiesToClassesConverter, self).__init__(thresholds)
