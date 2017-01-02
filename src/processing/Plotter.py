import matplotlib.pyplot as plt

class Plotter:


    def plot(self, sensors_data):
        figure = plt.figure()
        counter = 0
        for sensor in sensors_data:
            counter += 1
            subplot = figure.add_subplot(4, 4, counter)
            subplot.set_title(sensor)
            values = sensors_data[sensor]
            subplot.bar(range(1,len(values)), values, alpha=0.5)

        plt.show()

