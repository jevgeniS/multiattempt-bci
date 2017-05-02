class constants:
    # Sensor names in Plot and Extraction tab
    REQUIRED_PROBABILITY = 0.99
    SENSORS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")

    HEADSET_FREQ = 128

    EEG_MIN_FREQ = 1
    EEG_MAX_FREQ = 45

    USER_TRAINING_DURATION_S = 10
    USER_TEST_SAMPLES = 9
    MATH_QUIZ_DIGITS_START = 10000
    MATH_QUIZ_DIGITS_END = 999999
    TRAINING_DATA_PERCENTAGE = 70

    SAMPLES_PER_TEST_SESSION= USER_TRAINING_DURATION_S*2-2


    #DATA_FILE = "dataset2017-04-04 090820.csv"
    DATA_FILE = "dataset aggregated.csv"
    #TEST_DATA_FILE = "dataset_test 2017-04-04 090820.csv"
    TEST_DATA_FILE = "dataset_test aggregated.csv"

    DATA_FILE_PREFIX = "dataset"
    DATA_FILE_EXT = ".csv"
    AMPLITUDE_VALUE_DIGITS_AFTER_ZERO = 3

    TARGETS = {"1": "Excitement", "2": "Relax"}
    APP_MODES = {"1": "Learning mode", "2": "Training mode", "3": "Real-time testing mode", "4": "Final test"}
    BACK_KEY = "B"

    CSV_DELIMITER = ';'

    RF_CLF_TREES=100