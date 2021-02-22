import os
import os.path
import logging
import torch

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

def localCost(small, big, offset_x, offset_y, local_size=5):
    s_ch, ssize_x, ssize_y = small.size()
    b_ch, bsize_x, bsize_y = big.size()

    s_size = ssize_x * ssize_y

    half_local_size = local_size // 2
    ix = int(round(bsize_x / 2 - ssize_x / 2 + offset_x - half_local_size))
    iy = int(round(bsize_y / 2 - ssize_y / 2 + offset_y - half_local_size))
    c = torch.zeros((local_size, local_size)).cuda()
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    for y in range(0, local_size):
        for x in range(0, local_size):
            # todo: weight towards patch center...
            if ssize_y != 68 or ssize_x != 68 or iy < 0 or ix < 0 or iy + ssize_y > bsize_y or ix + ssize_x > bsize_x :
                print("burf")
            d = small - big[:, iy:iy + ssize_y, ix:ix + ssize_x]
            s = d.abs().sum() / s_size
            c[y, x] = s
            if s.item() < c[min_y, min_x]:
                min_x = x
                min_y = y
            if s.item() > c[max_y, max_x]:
                max_x = x
                max_y = y
            ix += 1
        iy += 1
    return c, [min_x,min_y], [max_x, max_y]