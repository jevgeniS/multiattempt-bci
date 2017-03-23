import operator
import random


class MathQuizGenerator(object):
    OPERATIONS = [
        (operator.add, "+"),
        #(operator.mul, "*"),
        (operator.sub, "-")
    ]

    def generate(self):
        arguments = 1
        quiz = ""
        quiz +=  str(random.randint(1000, 10000))+" "
        for i in range(arguments):
            op, symbol = random.choice(MathQuizGenerator.OPERATIONS)
            quiz += str(symbol)+" "
            quiz += str(random.randint(1000, 10000))+" "
        return quiz
