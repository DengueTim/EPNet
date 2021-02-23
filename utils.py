import os
import os.path
import logging
import random
import numpy as np
from itertools import groupby

def logger(name):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s %(filename)s:%(lineno)s %(threadName)s] %(levelname)-8s %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    )
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    return logger

    #
    # file_handle_name = "file"
    # if file_handle_name in [h.name for h in logger.handlers]:
    #     return
    # if os.path.dirname(filepath) is not '':
    #     if not os.path.isdir(os.path.dirname(filepath)):
    #         os.makedirs(os.path.dirname(filepath))
    # file_handle = logging.FileHandler(filename=filepath, mode="a")
    # file_handle.set_name(file_handle_name)
    # file_handle.setFormatter(file_formatter)
    # logger.addHandler(file_handle)


class LossMonitor():
    def __init__(self):
        self.last = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, loss):
        self.last = loss
        self.sum += loss
        self.count +=1

    def average(self):
        return self.sum / self.count

def getImageFilenamesWithPaths(dir, filename_extention='.png'):
    filenames = [filename for filename in os.listdir(dir) if filename.endswith(filename_extention)]
    return [os.path.join(dir, filename) for filename in filenames]

def localCost(small, big, offset_x, offset_y, std_dev=8, n_samples = 300):
    s_ch, ssize_x, ssize_y = small.size()
    b_ch, bsize_x, bsize_y = big.size()

    csize_y = bsize_y - ssize_y
    csize_x = bsize_x - ssize_x

    s_size = ssize_x * ssize_y

    offset_0x = csize_x // 2 + offset_x
    offset_0y = csize_y // 2 + offset_y

    xs = np.random.normal(offset_0x, std_dev, n_samples)
    ys = np.random.normal(offset_0y, std_dev, n_samples)
    xys = np.zeros((n_samples, 2), dtype=int)
    for i in range(n_samples):
        xys[i,0] = round(min(max(xs[i], 0), csize_x - 1))
        xys[i,1] = round(min(max(ys[i], 0), csize_y - 1))
    xys.sort(axis=0)

    costs = np.zeros((csize_y, csize_x))
    min_x = xys[0,0]
    min_y = xys[0,1]
    max_x = min_x
    max_y = min_y
    last_xy = [-1,-1]
    for (x, y) in xys:
        if last_xy == [x, y]:
            continue
        last_xy = [x,y]
        # todo: weight towards patch center...
        cost = small - big[:, y:y + ssize_y, x:x + ssize_x]
        cost = cost.abs().sum() / s_size
        costs[y, x] = cost
        if cost < costs[min_y, min_x]:
            min_x = x
            min_y = y
        if cost > costs[max_y, max_x]:
            max_x = x
            max_y = y

    return costs, \
           (min_x.item() - csize_x // 2, min_y.item() - csize_y // 2),\
           (max_x.item() - csize_x // 2, max_y.item() - csize_y // 2)