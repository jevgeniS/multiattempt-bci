import csv

import datetime
import numpy as np

from src import constants
from src.util.TimeStampGenerator import ts_format


class DataStorer(object):
    @staticmethod
    def file_path():
        return "datasets/" + constants.DATA_FILE

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




