class constants:
    # Sensor names in Plot and Extraction tab
    SENSORS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")

    HEADSET_FREQ = 128

    EEG_MIN_FREQ = 1
    EEG_MAX_FREQ = 45

    USER_TRAINING_DURATION_S = 10
    TRAINING_DATA_PERCENTAGE = 70

    SAMPLES_PER_TEST_SESSION= USER_TRAINING_DURATION_S*2-2


    DATA_FILE = "dataset2017-03-22 114758.csv"
    DATA_FILE_PREFIX = "dataset"
    DATA_FILE_EXT = ".csv"
    AMPLITUDE_VALUE_DIGITS_AFTER_ZERO = 3

    TARGETS = {"1": "Excitement", "2": "Relax"}
    BACK_KEY = "B"

    CSV_DELIMITER = ';'