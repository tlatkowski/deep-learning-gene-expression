import logging

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlotDecorator:

    def __init__(self, name):
        self.__name = name

    def __call__(self, f, *args, **kwargs):
        def new_f(*args, **kwargs):
            params, cost = f(*args, **kwargs)
            plt.ion()
            plt.plot(cost)
            plt.ylabel('some numbers')
            # plt.draw()
            plt.show()
            return params
        new_f.__name__ = f.__name__
        return new_f
