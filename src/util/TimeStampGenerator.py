import datetime
import time

def get_timestamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime(ts_format())

def get_timestamp_with_ms():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def ts_format():
    return '%Y-%m-%d %H:%M:%S'