import gevent
from emokit.emotiv import Emotiv


class Reader(object):

    @staticmethod
    def read(queue):
        print "Started"
        with Emotiv(write_values=False) as headset:
            while True:
                packet = headset.dequeue()
                if packet is not None and queue is not None:
                    queue.put(packet)
                #gevent.sleep(0.001)
