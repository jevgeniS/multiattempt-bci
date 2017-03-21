import csv

import datetime
import numpy as np

from src import constants
from src.util.TimeStampGenerator import ts_format, get_timestamp, get_timestamp_for_filename


class DataStorer():

    f_name = constants.DATA_FILE

    @staticmethod
    def file_path():
        return "datasets/" + DataStorer.f_name

    @staticmethod
    def store(rows, timestamp):
        #TODO: writer leaves empty first line in the file
        number_format = "%."+str(constants.AMPLITUDE_VALUE_DIGITS_AFTER_ZERO)+"f"
        with open(DataStorer.file_path(), 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=constants.CSV_DELIMITER)
            for row in rows:
                row = map(lambda x: ((number_format % x) if isinstance(x, float) else x), row)
                row.insert(0, timestamp)
                writer.writerow(row)


    @staticmethod
    def read():
        with open(DataStorer.file_path(), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=constants.CSV_DELIMITER)
            rows = []
            for row in reader:
                rows.append(row[1:])
            print 'Samples read from file:'+str(len(rows))
            return rows

    @staticmethod
    def read_interval(start_date, end_date):
        with open(DataStorer.file_path(), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=constants.CSV_DELIMITER)
            rows = []
            for row in reader:
                timestamp = datetime.datetime.strptime(row[0], ts_format())
                if start_date <= timestamp < end_date:
                    rows.append(row[1:])
            print 'Samples read from file:'+str(len(rows))
            return np.array(rows)

    @staticmethod
    def select_file():
        answer = raw_input("To create a new file type '1', to use default '2'")
        if answer == "1":
            DataStorer.f_name = DataStorer().generate_file_name()
        else:
            DataStorer.f_name = constants.DATA_FILE
        print "Using file: " + DataStorer.f_name


    @staticmethod
    def generate_file_name():
        return str(constants.DATA_FILE_PREFIX + get_timestamp_for_filename()+constants.DATA_FILE_EXT)




