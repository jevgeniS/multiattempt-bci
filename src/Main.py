from LearningService import LearningService

if __name__ == "__main__":
    print "This is BCI application for prediction user thoughts"
    LearningService().start()

    answer = raw_input("If you want to continue with learning mode please type 'Y'")
    if answer == "Y":
        print "Switched to learning mode"
        LearningService().start()
    else:
        print "Switched to test mode"


