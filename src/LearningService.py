import time
import winsound
from multiprocessing import Process, Queue

from processing.RawDataTransformer import RawDataTransformer
from util.DataStorer import DataStorer
from util.ExitServiceException import ExitServiceException
from util.MathQuizGenerator import MathQuizGenerator
from util.TimeStampGenerator import get_timestamp

from constants import constants as c

from emotiv.Reader import Reader
from emotiv.Sampler import Sampler


def emotiv_start_reader(queue):
    Reader.read(queue)


class ExitService(object):
    pass


class LearningService(object):
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


    def start(self):
        target = self.select_target()

        print "Please concentrate on the '"+target+"' for " + str(c.USER_TRAINING_DURATION_S) + "s :"
        exercise=MathQuizGenerator().generate()

        window=self.draw_fix_cross(target+": "+str(c.USER_TRAINING_DURATION_S)+"s", exercise)
        time.sleep(2)
        packet_queue = Queue()
        process = Process(target=emotiv_start_reader, args=(packet_queue,))
        process.start()
        raw_data = Sampler(packet_queue).get_samples(c.USER_TRAINING_DURATION_S)
        process.terminate()
        freq_domains = RawDataTransformer(raw_data).transform()
        windows_number = len(freq_domains.values()[0])
        ts = get_timestamp()
        samples = []
        for w in range(windows_number):
            sample = [target]
            for sensor in freq_domains:
                sample += freq_domains[sensor][w]
            samples.append(sample)
        window.close()
        winsound.MessageBeep()
        answer = raw_input("Type X do skip save of current try")

        if answer != 'X':
            DataStorer.store(samples, ts)


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






