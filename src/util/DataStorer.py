import csv

from src import constants


class DataStorer(object):

    @staticmethod
    def store(rows, timestamp):
        #TODO: writer leaves empty first line in the file
        with open(constants.DATA_FILE, 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=constants.CSV_DELIMITER)
            for row in rows:
                row.insert(0, timestamp)
                writer.writerow(row)


    @staticmethod
    def read():
        with open(constants.DATA_FILE, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=constants.CSV_DELIMITER)
            rows = []
            for row in reader:
                rows.append(row)

            return rows


