import operator
import random

from constants import constants


class MathQuizGenerator(object):
    OPERATIONS = [
        (operator.add, "+"),
        #(operator.mul, "*"),
        (operator.sub, "-")
    ]

    def generate(self):
        arguments = 1
        quiz = ""
        quiz +=  str(random.randint(constants.MATH_QUIZ_DIGITS_START, constants.MATH_QUIZ_DIGITS_END))+" "
        for i in range(arguments):
            op, symbol = random.choice(MathQuizGenerator.OPERATIONS)
            quiz += str(symbol)+" "
            quiz += str(random.randint(constants.MATH_QUIZ_DIGITS_START, constants.MATH_QUIZ_DIGITS_END))+" "
        return quiz
