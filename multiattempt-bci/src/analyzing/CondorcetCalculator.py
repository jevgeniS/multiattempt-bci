import math


class CondorcetCalculator(object):

    def calculate_number_of_voters(self, baseline_p, required_p):

        if baseline_p <= 0.50:
            raise Exception("Baseline probability should be positive")

        p = 0
        N=0
        while p < required_p:
            N += 1
            p = 0

            if N%2==0:
                majority_number = N / 2 + 1
            else:
                majority_number = int(math.ceil(N/2.0))

            for k in range(majority_number, N+1):
                p += self.nCr(N, k) * baseline_p**k * (1-baseline_p)**(N-k)
        return (N,p)


    def nCr(self, n, k):
        f = math.factorial
        return f(n) / (f(k) * f(n - k))


    def calculate_prob_for_voters(self, baseline_p, voters):

        if baseline_p <= 0.50:
            raise Exception("Baseline probability should be positive")

        p = 0
        N=0
        while N < voters:
            N += 1
            p = 0

            if N%2==0:
                majority_number = N / 2 + 1
            else:
                majority_number = int(math.ceil(N/2.0))

            for k in range(majority_number, N+1):
                p += self.nCr(N, k) * baseline_p**k * (1-baseline_p)**(N-k)
        return (N,p)
