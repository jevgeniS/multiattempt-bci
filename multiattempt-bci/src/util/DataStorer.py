import csv

import datetime
import numpy as np

from constants import constants
from util.TimeStampGenerator import ts_format, get_timestamp, get_timestamp_for_filename


class DataStorer(object):

    def __init__(self, f_name):
        self.f_name = f_name

    def file_path(self):
        return "../../datasets/" + self.f_name

    def store(self, rows, timestamp):
        #TODO: writer leaves empty first line in the file
        number_format = "%."+str(constants.AMPLITUDE_VALUE_DIGITS_AFTER_ZERO)+"f"
        with open(self.file_path(), 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=constants.CSV_DELIMITER)
            for row in rows:
                row = map(lambda x: ((number_format % x) if isinstance(x, float) else x), row)
                row.insert(0, timestamp)
                writer.writerow(row)

    def read(self):
        path = self.file_path()
        with open(path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=constants.CSV_DELIMITER)
            rows = []
            for row in reader:
                rows.append(row[1:])
            print 'Samples read from file:'+str(len(rows))
            return rows

    def read_with_timestamps(self):
        path = self.file_path()
        with open(path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=constants.CSV_DELIMITER)
            rows = []
            for row in reader:
                rows.append(row)
            print 'Samples read from file:'+str(len(rows))
            return rows


    def read_interval(self, start_date, end_date):
        with open(self.file_path(), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=constants.CSV_DELIMITER)
            rows = []
            for row in reader:
                timestamp = datetime.datetime.strptime(row[0], ts_format())
                if start_date <= timestamp < end_date:
                    rows.append(row[1:])
            print 'Samples read from file:'+str(len(rows))
            return np.array(rows)

    def select_file(self, test_data=False):
        if test_data:
            self.f_name = constants.TEST_DATA_FILE
        else:
            answer = raw_input("To create a new file type '1', to use default '2'")
            if answer == "1":
                DataStorer.f_name = DataStorer().generate_file_name()
            else:
                DataStorer.f_name = constants.DATA_FILE
        print "Using file: " + DataStorer.f_name


    def generate_file_name(self):
        return str(constants.DATA_FILE_PREFIX + get_timestamp_for_filename()+constants.DATA_FILE_EXT)




