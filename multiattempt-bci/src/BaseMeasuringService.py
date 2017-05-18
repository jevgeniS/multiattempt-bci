import time
import winsound
from multiprocessing import Process, Queue

from constants import constants as c
from emotiv.Reader import Reader
from emotiv.Sampler import Sampler
from processing.RawDataTransformer import RawDataTransformer
from util.DataStorer import DataStorer
from util.ExitServiceException import ExitServiceException
from util.MathQuizGenerator import MathQuizGenerator


def emotiv_start_reader(queue):
    Reader.read(queue)


class ExitService(object):
    pass


class BaseMeasuringService(object):

    def __init__(self):
        self.train_data_storer=self.create_traindata_storer()

    def measure(self):
        i = 1
        self.train_data_storer.select_file()
        try:
            while (True):
                print "Try number:" + str(i)
                self.start()
                i += 1
        except ExitServiceException as e:
            pass
            #TrainingService().train()

    def select_target(self):
        print "Select target typing the number of a target"
        for key in c.TARGETS.keys():
            print key + ":" + c.TARGETS[key]
        print
        print c.BACK_KEY+":Exit Learning"
        answer = raw_input()

        if (answer == c.BACK_KEY):
            raise ExitServiceException()

        return c.TARGETS[answer]

    def get_number_of_samples_per_session(self):
        raise NotImplementedError("Please Implement this method")

    def start(self):
        target = self.select_target()
        sessions_amount = self.get_number_of_samples_per_session()
        duration_s = sessions_amount*c.USER_TRAINING_DURATION_S
        print "Please concentrate on the '"+target+"' for " + str(duration_s) + "s :"
        exercise = MathQuizGenerator().generate()

        window=self.draw_fix_cross(target+": "+str(duration_s)+"s", exercise)
        time.sleep(2)
        packet_queue = Queue()
        process = Process(target=emotiv_start_reader, args=(packet_queue,))
        process.start()
        raw_data = Sampler(packet_queue).get_samples(duration_s)
        process.terminate()
        window.close()
        winsound.MessageBeep()
        samples = []
        chunk_size = len(raw_data)/sessions_amount
        for i in range(sessions_amount):
            chunk = raw_data[(chunk_size*i):(chunk_size*i+chunk_size)]
            freq_domains = RawDataTransformer(chunk).transform()
            windows_number = len(freq_domains.values()[0])

            for w in range(windows_number):
                sample = [target]
                for sensor in freq_domains:
                    sample += freq_domains[sensor][w]
                samples.append(sample)

        self.process_prediction_data(samples)



    def process_prediction_data(self, samples):
        raise NotImplementedError("Please Implement this method in child class")

    def draw_fix_cross(self, target, text):
        from psychopy import visual

        mywin = visual.Window([800, 600], monitor="testMonitor", units="deg")

        fixation = visual.GratingStim(win=mywin, mask="cross", size=0.4, pos=[0, 0], sf=0, color="red")
        target_stim = visual.TextStim(mywin, target, pos=(0, 8), colorSpace='rgb')
        text_stim = visual.TextStim(mywin, text, pos=(0, -0.5), colorSpace='rgb')

        target_stim.draw()
        text_stim.draw()
        fixation.draw()
        mywin.update()
        mywin.winHandle.activate()

        return mywin

    def create_traindata_storer(self):
        storer = DataStorer(None)
        return storer






