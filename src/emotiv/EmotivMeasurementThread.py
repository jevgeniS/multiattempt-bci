import threading

from emotiv.MyEmotiv import MyEmotiv


class EmotivMeasurementThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.emotiv = MyEmotiv()
        self.emotiv.connect()
        self.result = []

    def run(self):
        while True:
            self.measure()

    def measure(self):
        self.result.append(self.emotiv.getPacket())

    def stop(self):
        self.emotiv.cleanUp()
        return self.result

